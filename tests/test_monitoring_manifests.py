"""Static checks that the monitoring stack matches what the API actually exports."""

import json
import re

import pytest
import yaml
from fastapi.testclient import TestClient

from heartrisk import api, config
from heartrisk.schemas import EXAMPLE_PATIENT

MON = config.PROJECT_ROOT / "k8s" / "monitoring"
DASHBOARD = MON / "grafana" / "dashboards" / "heartrisk-api.json"


@pytest.fixture(scope="module")
def exported_metrics() -> set[str]:
    """Metric family names the running API exposes on /metrics."""
    with TestClient(api.app) as c:
        c.post("/predict", json=EXAMPLE_PATIENT)
        text = c.get("/metrics").text
    names = {ln.split()[2] for ln in text.splitlines() if ln.startswith("# TYPE ")}
    suffixes = ("_bucket", "_count", "_sum", "_total", "_created")
    return names | {n + s for n in names for s in suffixes}


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD.read_text())


def test_kustomization_references_existing_files():
    kust = yaml.safe_load((MON / "kustomization.yaml").read_text())
    for res in kust["resources"]:
        assert (MON / res).exists(), res
    for gen in kust["configMapGenerator"]:
        for f in gen["files"]:
            assert (MON / f).exists(), f


def test_dashboard_queries_use_exported_metrics(dashboard, exported_metrics):
    for panel in dashboard["panels"]:
        for t in panel["targets"]:
            for metric in set(re.findall(r"\bheartrisk_[a-z_]+", t["expr"])):
                assert metric in exported_metrics, f"{panel['title']}: unknown metric {metric}"


def test_dashboard_panels_are_unique_and_use_provisioned_datasource(dashboard):
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids))
    ds = yaml.safe_load((MON / "grafana" / "datasources.yaml").read_text())["datasources"][0]
    assert all(p["datasource"]["uid"] == ds["uid"] for p in dashboard["panels"])


def test_prometheus_scrape_annotations_match_deployment():
    prom = yaml.safe_load((MON / "prometheus" / "prometheus.yml").read_text())
    job = next(j for j in prom["scrape_configs"] if j["job_name"] == "heartrisk-api")
    used = {lbl for rc in job["relabel_configs"] for lbl in rc.get("source_labels", [])}
    ann = yaml.safe_load((config.PROJECT_ROOT / "k8s" / "deployment.yaml").read_text())
    ann = ann["spec"]["template"]["metadata"]["annotations"]
    for key in ann:
        sd_label = "__meta_kubernetes_pod_annotation_" + re.sub(r"[^a-zA-Z0-9]", "_", key)
        assert sd_label in used, f"annotation {key} not used by Prometheus"
    assert job["kubernetes_sd_configs"][0]["namespaces"]["names"] == ["heartrisk"]


def test_alert_rules_use_exported_metrics(exported_metrics):
    rules = yaml.safe_load((MON / "prometheus" / "rules.yml").read_text())["groups"][0]["rules"]
    assert {r["alert"] for r in rules if "alert" in r} >= {"HeartRiskApiDown", "HeartRiskApiHighErrorRate"}
    for r in rules:
        for metric in set(re.findall(r"\bheartrisk_[a-z_]+", r["expr"])):
            assert metric in exported_metrics, metric


def test_monitoring_ingress_paths_match_sub_path_config():
    ing = yaml.safe_load((MON / "ingress.yaml").read_text())
    paths = {p["path"] for r in ing["spec"]["rules"] for p in r["http"]["paths"]}
    assert paths == {"/prometheus", "/grafana"}
    prom = (MON / "prometheus.yaml").read_text()
    graf = (MON / "grafana.yaml").read_text()
    assert "--web.route-prefix=/prometheus" in prom
    assert "/grafana/" in graf and 'GF_SERVER_SERVE_FROM_SUB_PATH, value: "true"' in graf
