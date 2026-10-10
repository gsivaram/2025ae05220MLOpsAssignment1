"""Run the whole pipeline from a clean checkout and record proof of execution.

Usage (from the repo root, with the virtual environment active):
    python scripts/run_pipeline.py                 # all steps
    python scripts/run_pipeline.py --from train    # resume from a step
    python scripts/run_pipeline.py --only test     # one step

Each step's output is streamed to the console and saved to reports/run/<step>.log;
reports/run/run_summary.md records host, OS, Python, package versions, timings
and pass/fail for every step. The run stops at the first failing step with a
non-zero exit code.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "reports" / "run"
PY = sys.executable

STEPS: list[tuple[str, str, list[str]]] = [
    ("lint", "Static checks (ruff)", [PY, "-m", "ruff", "check", "src", "scripts", "tests"]),
    ("data", "Download + clean dataset", [PY, "scripts/download_data.py", "--force"]),
    ("eda", "Execute EDA notebook", [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                                     "--inplace", "notebooks/01_eda.ipynb"]),
    ("train", "Train + tune + log to MLflow", [PY, "scripts/train.py"]),
    ("test", "Unit tests + coverage", [PY, "-m", "pytest", "--cov=heartrisk", "--cov-report=term"]),
    ("predict", "Inference on sample request", [PY, "scripts/predict.py", "sample_request.json"]),
]
STEP_NAMES = [s[0] for s in STEPS]
PACKAGES = ["numpy", "pandas", "scikit-learn", "mlflow", "matplotlib", "fastapi", "pytest"]


def environment() -> dict[str, str]:
    def version(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "not installed"

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "Host": socket.gethostname(),
        "OS": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Python": f"{platform.python_version()} at {PY}",
        "Git commit": commit,
        "Started": datetime.now().astimezone().isoformat(timespec="seconds"),
        **{f"pkg {p}": version(p) for p in PACKAGES},
    }


def run_step(name: str, title: str, cmd: list[str]) -> tuple[bool, float]:
    print(f"\n{'=' * 78}\n>> [{name}] {title}\n  $ {' '.join(Path(c).name if c == PY else c for c in cmd)}\n{'=' * 78}",
          flush=True)
    log_path = RUN_DIR / f"{name}.log"
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        # UTF-8 everywhere so Windows consoles don't choke on child-process output, and this
        # interpreter's bin/Scripts dir first on PATH so Jupyter kernels resolve to the same venv.
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                     "PATH": os.pathsep.join([str(Path(PY).parent), os.environ.get("PATH", "")])}
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=child_env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    status = "passed" if ok else f"FAILED (exit {proc.returncode})"
    print(f"{'[OK]' if ok else '[FAIL]'} [{name}] {status} in {elapsed:.1f}s - log: {log_path.relative_to(ROOT)}",
          flush=True)
    return ok, elapsed


def write_summary(env: dict[str, str], results: list[tuple[str, str, bool, float]]) -> Path:
    lines = ["# Pipeline run summary", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in env.items()]
    lines += ["", "| Step | Description | Result | Time |", "|---|---|---|---|"]
    lines += [f"| {n} | {t} | {'PASS' if ok else '**FAIL**'} | {s:.1f}s |" for n, t, ok, s in results]
    path = RUN_DIR / "run_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=STEP_NAMES, help="start at this step")
    group.add_argument("--only", choices=STEP_NAMES, help="run just this step")
    args = parser.parse_args()

    if sys.version_info < (3, 10):  # noqa: UP036 - friendly message for old lab interpreters
        print(f"Python 3.10+ required, found {platform.python_version()}", file=sys.stderr)
        return 2

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        steps = STEPS[STEP_NAMES.index(args.start):]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env = environment()
    print("Environment:\n" + "\n".join(f"  {k:<18} {v}" for k, v in env.items()))

    results = []
    for name, title, cmd in steps:
        ok, elapsed = run_step(name, title, cmd)
        results.append((name, title, ok, elapsed))
        if not ok:
            break

    summary = write_summary(env, results)
    failed = [r for r in results if not r[2]]
    print(f"\n{'=' * 78}")
    print(f"{'PIPELINE FAILED at step ' + failed[0][0] if failed else 'PIPELINE PASSED'} - "
          f"summary: {summary.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
