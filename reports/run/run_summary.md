# Pipeline run summary

| | |
|---|---|
| Host | ip-10-3-21-203.ap-south-1.compute.internal |
| OS | Linux 5.14.0-503.14.1.el9_5.x86_64 (x86_64) |
| Python | 3.12.9 at /home/s2025ae05220/workspace/2025ae05220MLOpsAssignment1/.venv/bin/python |
| Git commit | 01ff095 |
| Docker | 28.0.1 |
| Started | 2026-10-17T13:20:51+05:30 |
| pkg numpy | 2.5.3 |
| pkg pandas | 2.3.3 |
| pkg scikit-learn | 1.7.2 |
| pkg mlflow | 2.22.5 |
| pkg matplotlib | 3.11.2 |
| pkg fastapi | 0.143.0 |
| pkg pytest | 9.1.1 |

| Step | Description | Result | Time |
|---|---|---|---|
| lint | Static checks (ruff) | PASS | 0.9s |
| data | Download + clean dataset | PASS | 11.1s |
| eda | Execute EDA notebook | PASS | 42.4s |
| train | Train + tune + log to MLflow | PASS | 254.9s |
| test | Unit tests + coverage | PASS | 108.9s |
| predict | Inference on sample request | PASS | 12.1s |
| docker | Build image + container smoke test | PASS | 73.8s |
