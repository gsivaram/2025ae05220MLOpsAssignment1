#!/usr/bin/env bash
# Deploy the Heart Disease Risk API to a local Minikube cluster behind an NGINX Ingress.
#
# Usage: scripts/k8s_deploy.sh [command]
#   up       (default) install tools if missing -> start minikube -> build + load image
#            -> apply manifests -> wait for rollout -> verify through the Ingress
#   monitoring  deploy Prometheus + Grafana (k8s/monitoring) and verify scraping + dashboard
#   all      up + monitoring
#   verify   re-run the endpoint checks against the running deployment
#   status   show cluster objects
#   down     delete the heartrisk namespace (cluster keeps running)
#   destroy  delete the whole minikube profile
#
# No sudo needed: minikube/kubectl are installed into ~/.local/bin when absent.
set -euo pipefail

MINIKUBE_VERSION="${MINIKUBE_VERSION:-v1.39.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-v1.37.1}"
PROFILE="${PROFILE:-heartrisk}"
NAMESPACE="heartrisk"
HOST="heartrisk.local"
IMAGE_REPO="heartrisk-api"
CPUS="${MINIKUBE_CPUS:-2}"
MEMORY="${MINIKUBE_MEMORY:-3072}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
TOOLS_DIR="${HOME}/.local/bin"

cd "$(dirname "$0")/.."
export PATH="${TOOLS_DIR}:${PATH}"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }
fail() { printf '\n\033[31mFAIL: %s\033[0m\n' "$*"; exit 1; }
k()    { kubectl --context "$PROFILE" "$@"; }

platform() {
  case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=darwin ;; *) fail "unsupported OS $(uname -s)" ;; esac
  case "$(uname -m)" in x86_64|amd64) ARCH=amd64 ;; arm64|aarch64) ARCH=arm64 ;; *) fail "unsupported arch $(uname -m)" ;; esac
}

install_tools() {
  step "Tools (pinned: minikube ${MINIKUBE_VERSION}, kubectl ${KUBECTL_VERSION})"
  platform
  mkdir -p "$TOOLS_DIR"
  # A different system-wide version is left alone; the pinned one goes into ~/.local/bin,
  # which is first on PATH, so every machine runs the same minikube/kubectl/Kubernetes.
  local have
  have="$(minikube version --short 2>/dev/null || true)"
  if [ "$have" != "$MINIKUBE_VERSION" ]; then
    echo "  installing minikube ${MINIKUBE_VERSION} (${OS}/${ARCH}) into ${TOOLS_DIR} (found: ${have:-none})"
    curl -fsSLo "${TOOLS_DIR}/minikube" \
      "https://github.com/kubernetes/minikube/releases/download/${MINIKUBE_VERSION}/minikube-${OS}-${ARCH}"
    chmod +x "${TOOLS_DIR}/minikube"
    hash -r
  fi
  have="$(kubectl version --client -o json 2>/dev/null | grep -m1 gitVersion | cut -d'"' -f4 || true)"
  if [ "$have" != "$KUBECTL_VERSION" ]; then
    echo "  installing kubectl ${KUBECTL_VERSION} (${OS}/${ARCH}) into ${TOOLS_DIR} (found: ${have:-none})"
    curl -fsSLo "${TOOLS_DIR}/kubectl" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"
    chmod +x "${TOOLS_DIR}/kubectl"
    hash -r
  fi
  docker info >/dev/null 2>&1 || fail "Docker is not running / not accessible without sudo"
  ok "docker $(docker version --format '{{.Server.Version}}')"
  ok "$(minikube version --short)"
  ok "kubectl $(kubectl version --client -o json | grep -m1 gitVersion | cut -d'"' -f4)"
}

start_cluster() {
  step "Minikube cluster '${PROFILE}'"
  local want_k8s running_k8s
  want_k8s="$(minikube config defaults kubernetes-version 2>/dev/null | head -1 | tr -d '* ' || true)"
  running_k8s="$(k get nodes -o jsonpath='{.items[0].status.nodeInfo.kubeletVersion}' 2>/dev/null || true)"
  if [ -n "$running_k8s" ] && [ -n "$want_k8s" ] && [ "$running_k8s" != "$want_k8s" ]; then
    echo "  cluster runs Kubernetes ${running_k8s} but minikube ${MINIKUBE_VERSION} expects ${want_k8s}: recreating"
    minikube delete -p "$PROFILE"
  fi
  if minikube -p "$PROFILE" status >/dev/null 2>&1; then
    ok "already running"
  else
    minikube start -p "$PROFILE" --driver=docker --cpus="$CPUS" --memory="$MEMORY" \
      --addons=ingress --addons=metrics-server
  fi
  minikube -p "$PROFILE" addons enable ingress >/dev/null
  minikube -p "$PROFILE" addons enable metrics-server >/dev/null
  echo "  waiting for the ingress controller..."
  k -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=300s
  ok "node $(k get nodes -o jsonpath='{.items[0].metadata.name}') (Kubernetes $(k get nodes -o jsonpath='{.items[0].status.nodeInfo.kubeletVersion}'))"
}

build_and_load() {
  step "Build image and load it into the cluster"
  [ -f models/model.joblib ] || fail "models/model.joblib missing - run scripts/train.py first"
  TAG="$(git rev-parse --short HEAD 2>/dev/null || echo local)-$(date +%Y%m%d%H%M%S)"
  IMAGE="${IMAGE_REPO}:${TAG}"
  docker build -q -t "$IMAGE" -t "${IMAGE_REPO}:latest" . >/dev/null
  ok "built ${IMAGE}"
  minikube -p "$PROFILE" image load "$IMAGE"
  ok "loaded into minikube"
}

deploy() {
  step "Apply manifests (kustomize k8s/ with image ${IMAGE})"
  # Render once with the exact tag so no pod ever tries to pull :latest from a registry.
  k kustomize k8s/ | sed "s#image: ${IMAGE_REPO}:latest#image: ${IMAGE}#" | k apply -f -
  k -n "$NAMESPACE" annotate deployment/heartrisk-api kubernetes.io/change-cause="deploy ${IMAGE}" --overwrite >/dev/null
  step "Rolling out ${IMAGE}"
  k -n "$NAMESPACE" rollout status deployment/heartrisk-api --timeout=300s
}

# Resolve how to reach the Ingress: the minikube IP directly (Linux), else a port-forward
# to the ingress controller (macOS/Windows, where the docker-driver IP is not routable).
resolve_endpoint() {
  local ip direct=no
  ip="$(minikube -p "$PROFILE" ip)"
  # Right after a rollout the ingress controller needs a few seconds to pick up the new
  # endpoints, so retry before deciding the IP is unreachable (macOS never succeeds here).
  for _ in $(seq 1 10); do
    if curl -fs -m 3 -H "Host: ${HOST}" "http://${ip}/health" >/dev/null 2>&1; then direct=yes; break; fi
    [ "$(uname -s)" = "Darwin" ] && break
    sleep 3
  done
  if [ "$direct" = yes ]; then
    BASE="http://${ip}"
    ok "Ingress reachable at ${BASE} (Host: ${HOST})"
  else
    k -n ingress-nginx port-forward svc/ingress-nginx-controller "${LOCAL_PORT}:80" >/dev/null 2>&1 &
    PF_PID=$!
    trap 'kill ${PF_PID} 2>/dev/null || true' EXIT
    for _ in $(seq 1 20); do
      curl -fs -m 2 -H "Host: ${HOST}" "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1 && break
      sleep 1
    done
    BASE="http://127.0.0.1:${LOCAL_PORT}"
    ok "minikube IP ${ip} not routable from this host; using port-forward to the ingress controller: ${BASE}"
  fi
}

verify() {
  step "Verify endpoints through the Ingress"
  resolve_endpoint
  local H=(-H "Host: ${HOST}")

  echo "  GET /health"
  curl -fsS "${H[@]}" "${BASE}/health" || fail "/health"
  echo

  echo "  POST /predict (sample_request.json)"
  curl -fsS "${H[@]}" -X POST "${BASE}/predict" -H 'Content-Type: application/json' -d @sample_request.json \
    | tee /dev/stderr | grep -q probability_disease || fail "/predict"
  echo

  echo "  POST /predict invalid input"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "${H[@]}" -X POST "${BASE}/predict" \
    -H 'Content-Type: application/json' -d '{"age": 630}')
  [ "$code" = "422" ] || fail "expected 422, got ${code}"
  ok "HTTP 422 for invalid input"

  echo "  Load balancing: 20 requests, replica that served each"
  for _ in $(seq 1 20); do
    curl -fsS -o /dev/null -D - "${H[@]}" "${BASE}/health" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-served-by"{print $2}'
  done | sort | uniq -c | sed 's/^/   /'

  echo "  GET /metrics (excerpt)"
  curl -fsS "${H[@]}" "${BASE}/metrics" | grep -E '^heartrisk_(http_requests_total|predictions_total)' | head -5

  printf '\n\033[32mKUBERNETES DEPLOYMENT VERIFIED\033[0m\n'
}

monitoring() {
  step "Apply monitoring stack (kubectl apply -k k8s/monitoring/)"
  k get namespace "$NAMESPACE" >/dev/null 2>&1 || fail "deploy the app first (scripts/k8s_deploy.sh up)"
  k apply -k k8s/monitoring/
  k -n monitoring rollout status deployment/prometheus --timeout=300s
  k -n monitoring rollout status deployment/grafana --timeout=300s
  verify_monitoring
}

# Prometheus HTTP API helper: prom <path-and-query>
prom() { curl -fsS -H "Host: ${HOST}" "${BASE}/prometheus/api/v1/$1"; }

verify_monitoring() {
  step "Verify monitoring through the Ingress"
  [ -n "${BASE:-}" ] || resolve_endpoint
  curl -fsS -o /dev/null -H "Host: ${HOST}" "${BASE}/prometheus/-/ready" || fail "Prometheus not ready"
  ok "Prometheus ready at ${BASE}/prometheus"

  local up=0
  for _ in $(seq 1 30); do
    up=$(prom 'targets?state=active' | python3 -c 'import json,sys
t=json.load(sys.stdin)["data"]["activeTargets"]
print(sum(1 for x in t if x["labels"].get("job")=="heartrisk-api" and x["health"]=="up"))')
    [ "$up" -ge 2 ] && break
    sleep 2
  done
  [ "$up" -ge 2 ] || fail "Prometheus scrapes ${up} API pods (expected >= 2)"
  ok "Prometheus scraping ${up} heartrisk-api pods"

  echo "  generating 20 s of traffic (scripts/load_test.py)..."
  python3 scripts/load_test.py --url "$BASE" --host "$HOST" --duration 20 --rps 8 | tail -6
  sleep 6   # one scrape interval + margin
  local rate
  rate=$(prom 'query?query=sum(rate(heartrisk_http_requests_total%5B1m%5D))' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(round(float(r[0]["value"][1]),2) if r else 0)')
  ok "Prometheus sees ${rate} req/s on the API"

  curl -fsS -H "Host: ${HOST}" "${BASE}/grafana/api/health" | grep -q '"database": *"ok"' || fail "Grafana unhealthy"
  local title
  title=$(curl -fsS -H "Host: ${HOST}" "${BASE}/grafana/api/dashboards/uid/heartrisk-api" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["dashboard"]["title"])') || fail "dashboard not provisioned"
  ok "Grafana healthy, dashboard provisioned: ${title}"

  printf '\n\033[32mMONITORING VERIFIED\033[0m\n'
  local ip; ip="$(minikube -p "$PROFILE" ip)"
  echo "  Grafana:    http://${ip}/grafana/      (anonymous view; admin / heartrisk-admin to edit)"
  echo "  Prometheus: http://${ip}/prometheus/   (Status > Targets, Alerts)"
  echo "  Traffic:    python3 scripts/load_test.py --url http://${ip} --duration 300 --rps 10"
}

status() {
  step "Cluster objects (namespace ${NAMESPACE})"
  k -n "$NAMESPACE" get deploy,rs,pods,svc,ingress,hpa,pdb -o wide
  if k get namespace monitoring >/dev/null 2>&1; then
    step "Monitoring objects (namespace monitoring)"
    k -n monitoring get deploy,pods,svc,ingress -o wide
  fi
  step "Rollout history"
  k -n "$NAMESPACE" rollout history deployment/heartrisk-api | tail -5
  step "Access"
  echo "  minikube IP: $(minikube -p "$PROFILE" ip)"
  echo "  curl -H 'Host: ${HOST}' http://$(minikube -p "$PROFILE" ip)/health        # Linux"
  echo "  browser: http://$(minikube -p "$PROFILE" ip)/docs                     # Linux (Swagger UI via Ingress)"
  echo "  macOS:   kubectl --context ${PROFILE} -n ingress-nginx port-forward svc/ingress-nginx-controller ${LOCAL_PORT}:80"
}

case "${1:-up}" in
  up)      install_tools; start_cluster; build_and_load; deploy; verify; status ;;
  monitoring) monitoring ;;
  all)     install_tools; start_cluster; build_and_load; deploy; verify; monitoring; status ;;
  verify)  verify ;;
  status)  status ;;
  down)    k delete namespace monitoring "$NAMESPACE" --ignore-not-found ;;
  destroy) minikube delete -p "$PROFILE" ;;
  *)       sed -n '2,15p' "$0"; exit 2 ;;
esac
