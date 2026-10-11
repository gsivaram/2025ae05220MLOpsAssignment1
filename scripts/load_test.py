"""Send realistic traffic to the API so the monitoring dashboard has live data.

Patients are sampled from the cleaned dataset (with small jitter); a share of requests
is deliberately invalid (-> HTTP 422) and some are batch calls. Standard library only.

Usage:
    python scripts/load_test.py --url http://192.168.49.2 --duration 120 --rps 10
    python scripts/load_test.py --url http://127.0.0.1:8080 --host heartrisk.local   # via port-forward
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "heart_clean.csv"
INT_FIELDS = {"age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
INVALID_EDITS = [{"age": 630}, {"cp": 9}, {"chol": -10}, {"sex": "male"}, {"thalach": None}]


def load_patients() -> list[dict]:
    patients = []
    with DATA.open() as f:
        for row in csv.DictReader(f):
            row.pop("target", None)
            patients.append({k: (None if v == "" else int(float(v)) if k in INT_FIELDS else float(v))
                             for k, v in row.items()})
    return patients


def jitter(p: dict, rng: random.Random) -> dict:
    q = dict(p)
    for k, spread in (("trestbps", 6), ("chol", 15), ("thalach", 8)):
        q[k] = round(q[k] + rng.uniform(-spread, spread), 1)
    q["oldpeak"] = round(max(0.0, q["oldpeak"] + rng.uniform(-0.3, 0.3)), 1)
    return q


def post(url: str, path: str, body: dict, host: str | None, timeout: float = 5) -> int:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"Host": host} if host else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="base URL of the API / Ingress")
    parser.add_argument("--host", default=None, help="Host header (e.g. heartrisk.local)")
    parser.add_argument("--duration", type=float, default=120, help="seconds to run")
    parser.add_argument("--rps", type=float, default=10, help="target requests per second")
    parser.add_argument("--invalid-ratio", type=float, default=0.1, help="share of deliberately invalid requests")
    parser.add_argument("--batch-ratio", type=float, default=0.05, help="share of /predict/batch calls")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patients = load_patients()
    counts: Counter = Counter()
    lock = threading.Lock()
    stop_at = time.monotonic() + args.duration
    interval = args.workers / args.rps

    def worker(n: int) -> None:
        rng = random.Random(args.seed + n)
        next_at = time.monotonic()
        while time.monotonic() < stop_at:
            roll = rng.random()
            if roll < args.invalid_ratio:
                path, body = "/predict", {**rng.choice(patients), **rng.choice(INVALID_EDITS)}
            elif roll < args.invalid_ratio + args.batch_ratio:
                path, body = "/predict/batch", {"patients": [jitter(rng.choice(patients), rng)
                                                             for _ in range(rng.randint(2, 10))]}
            else:
                path, body = "/predict", jitter(rng.choice(patients), rng)
            status = post(args.url, path, body, args.host)
            with lock:
                counts[(path, status)] += 1
            next_at += interval
            time.sleep(max(0.0, next_at - time.monotonic()))

    print(f"Sending ~{args.rps} req/s to {args.url} for {args.duration:.0f}s "
          f"({args.invalid_ratio:.0%} invalid, {args.batch_ratio:.0%} batch)...")
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.workers)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0

    total = sum(counts.values())
    print(f"\n{total} requests in {elapsed:.0f}s ({total / elapsed:.1f} req/s)")
    for (path, status), n in sorted(counts.items()):
        print(f"  {path:<16} HTTP {status or 'ERR':<4} {n}")
    if counts.get(("/predict", 0)) or counts.get(("/predict/batch", 0)):
        raise SystemExit("some requests failed to connect - is the URL right?")


if __name__ == "__main__":
    main()
