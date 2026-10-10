#!/usr/bin/env bash
# Build the API image, run it, and verify the endpoints with sample input.
# Usage: scripts/docker_smoke_test.sh [image-tag] [host-port]
# Exits non-zero (and prints container logs) if any check fails.
set -euo pipefail

IMAGE="${1:-heartrisk-api:latest}"
PORT="${2:-8000}"
NAME="heartrisk-smoke-$$"
BASE="http://127.0.0.1:${PORT}"
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAIL: %s\033[0m\n' "$*"; docker logs "$NAME" 2>&1 | tail -30 || true; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

step "Build ${IMAGE}"
docker build -t "$IMAGE" .
docker image ls "$IMAGE" --format 'image size: {{.Size}}'

step "Run container ${NAME} on port ${PORT}"
docker run -d --name "$NAME" -p "${PORT}:8000" "$IMAGE" >/dev/null

step "Wait for /health"
for i in $(seq 1 30); do
  if curl -fs "${BASE}/health" >/dev/null 2>&1; then break; fi
  [ "$i" -eq 30 ] && fail "API did not become healthy within 30 s"
  sleep 1
done
curl -fs "${BASE}/health"; echo

step "POST /predict with sample_request.json"
curl -fs -X POST "${BASE}/predict" -H 'Content-Type: application/json' -d @sample_request.json \
  | tee /dev/stderr | grep -q '"probability_disease"' || fail "/predict did not return a prediction"
echo

step "POST /predict with invalid input (expect HTTP 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BASE}/predict" -H 'Content-Type: application/json' \
  -d '{"age": 630}')
echo "HTTP ${code}"
[ "$code" = "422" ] || fail "expected 422 for invalid input, got ${code}"

step "GET /metrics (excerpt)"
curl -fs "${BASE}/metrics" | grep -E '^heartrisk_(http_requests_total|predictions_total|model_info)' \
  || fail "/metrics missing heartrisk_* series"

step "Docker HEALTHCHECK status (wait for healthy)"
for i in $(seq 1 30); do
  status=$(docker inspect --format '{{.State.Health.Status}}' "$NAME")
  [ "$status" = "healthy" ] && break
  [ "$i" -eq 30 ] && fail "container health is '${status}' after 30 s"
  sleep 1
done
echo "$status"

step "Request logs (JSON, last 5 lines)"
docker logs "$NAME" 2>&1 | tail -5

printf '\n\033[32mDOCKER SMOKE TEST PASSED\033[0m\n'
