"""Static checks that the Kubernetes manifests agree with the API and the Dockerfile."""

from pathlib import Path

import pytest
import yaml

from heartrisk import api, config

K8S = config.PROJECT_ROOT / "k8s"


def load(name: str) -> dict:
    return yaml.safe_load((K8S / name).read_text())


@pytest.fixture(scope="module")
def deployment():
    return load("deployment.yaml")


@pytest.fixture(scope="module")
def container(deployment):
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_kustomization_lists_every_manifest():
    listed = set(load("kustomization.yaml")["resources"])
    on_disk = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert listed == on_disk


def test_probes_target_a_real_api_route(container):
    routes = {r.path for r in api.app.routes}
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"]["path"] in routes


def test_container_port_matches_dockerfile(container):
    dockerfile = (config.PROJECT_ROOT / "Dockerfile").read_text()
    assert "EXPOSE 8000" in dockerfile and "--port\", \"8000\"" in dockerfile
    assert container["ports"][0]["containerPort"] == 8000


def test_service_selects_the_deployment_pods(deployment):
    service = load("service.yaml")
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= pod_labels.items()
    assert deployment["spec"]["selector"]["matchLabels"].items() <= pod_labels.items()


def test_ingress_routes_to_the_service():
    ingress, service = load("ingress.yaml"), load("service.yaml")
    rules = ingress["spec"]["rules"]
    assert any(r.get("host") == "heartrisk.local" for r in rules)
    for rule in rules:
        backend = rule["http"]["paths"][0]["backend"]["service"]
        assert backend["name"] == service["metadata"]["name"]
        assert backend["port"]["name"] in {p["name"] for p in service["spec"]["ports"]}


def test_hpa_and_pdb_target_the_deployment(deployment):
    hpa, pdb = load("hpa.yaml"), load("pdb.yaml")
    assert hpa["spec"]["scaleTargetRef"]["name"] == deployment["metadata"]["name"]
    assert hpa["spec"]["minReplicas"] <= deployment["spec"]["replicas"] <= hpa["spec"]["maxReplicas"]
    assert pdb["spec"]["minAvailable"] < deployment["spec"]["replicas"]


def test_container_is_hardened(deployment, container):
    pod_sc = deployment["spec"]["template"]["spec"]["securityContext"]
    assert pod_sc["runAsNonRoot"] is True and pod_sc["runAsUser"] == 10001  # same uid as the Dockerfile user
    sc = container["securityContext"]
    assert sc["readOnlyRootFilesystem"] and not sc["allowPrivilegeEscalation"]
    assert container["resources"]["limits"] and container["resources"]["requests"]


def test_prometheus_scrape_annotations_point_at_metrics(deployment):
    ann = deployment["spec"]["template"]["metadata"]["annotations"]
    assert ann["prometheus.io/scrape"] == "true"
    assert ann["prometheus.io/path"] in {r.path for r in api.app.routes}


def test_rolling_update_never_drops_capacity(deployment):
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0


def test_manifests_are_valid_yaml():
    for path in Path(K8S).glob("*.yaml"):
        assert yaml.safe_load(path.read_text()), path.name
