# Heart Disease Risk — End-to-End MLOps

[![CI](https://github.com/gsivaram/2025ae05220MLOpsAssignment1/actions/workflows/ci.yml/badge.svg)](https://github.com/gsivaram/2025ae05220MLOpsAssignment1/actions/workflows/ci.yml)

Binary classifier predicting presence of heart disease from 13 clinical features
(UCI Heart Disease, Cleveland subset), packaged as a monitored, containerised API.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python scripts/download_data.py          # data/raw -> data/processed/heart_clean.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

python scripts/train.py                  # grid search + MLflow logging, writes models/
python scripts/predict.py sample_request.json
mlflow ui --backend-store-uri ./mlruns   # http://127.0.0.1:5000
```

## Modelling

* **Pipeline** (`src/heartrisk/features.py`): `ClinicalFeatures` (optional engineered
  `hr_reserve = thalach / (220 − age)`) → median-impute + standard-scale numerics,
  most-frequent-impute + one-hot nominal codes, impute binaries → classifier.
  One pickled `Pipeline` = identical preprocessing at train and serve time.
* **Candidates**: Logistic Regression, Random Forest, Gradient Boosting — each tuned by
  `GridSearchCV` (stratified 5-fold, 116 configurations in total) on an 80 % training
  split. The engineered feature is itself a grid parameter.
* **Selection**: highest CV ROC-AUC; the 20 % test split is used only for reporting.

| Model | CV ROC-AUC | CV accuracy | CV recall | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** (C=0.3, L2, balanced) | **0.907 ± 0.019** | 0.855 | 0.801 | 0.961 |
| Random Forest | 0.901 ± 0.025 | 0.822 | 0.774 | 0.951 |
| Gradient Boosting | 0.887 ± 0.023 | 0.826 | 0.774 | 0.959 |

## Experiment tracking

MLflow experiment `heart-disease-risk`: one parent run per model (best params, CV
mean/std + test metrics, ROC / PR / confusion / permutation-importance plots, CV
table, logged model with signature and bundled code), a nested child run per grid
candidate, and a `model_selection` run with the comparison plots. The chosen run is
tagged `selected=true`.

## Model artefacts

`models/model.joblib` (full pipeline) and `models/metadata.json` (version, metrics,
params, features, threshold, MLflow run id). The MLflow copy under the run's
`model/` artifact also loads with `mlflow.pyfunc.load_model`.

## Repository layout

| Path | Contents |
|---|---|
| `src/heartrisk/` | Library code: config, data cleaning, EDA plots, features, training, evaluation |
| `scripts/` | CLI entry points: download, train, predict |
| `models/` | Selected model pipeline + metadata |
| `notebooks/` | Narrative notebooks (EDA, …) |
| `data/raw`, `data/processed` | Raw UCI file and cleaned CSV |
| `reports/figures/` | Generated plots |
| `tests/` | Unit tests |
| `screenshots/` | Evidence for the report |

## Dataset

UCI Machine Learning Repository, *Heart Disease* (id 45),
`processed.cleveland.data` — 303 rows. Downloaded automatically by
`scripts/download_data.py`; target `num` (0–4) is binarised to presence (1) / absence (0).

## Prediction API (FastAPI) and Docker

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model status (Docker HEALTHCHECK, Kubernetes probes) |
| `GET /model` | loaded model's metadata (version, metrics, params) |
| `POST /predict` | one patient → `prediction`, `label`, `probability_disease`, `confidence`, `risk_band` |
| `POST /predict/batch` | up to 100 patients |
| `GET /metrics` | Prometheus metrics (requests, latency, predictions by label, probability histogram) |
| `GET /docs` | Swagger UI |

Inputs are validated against clinical ranges (HTTP 422 on bad input); `ca` and `thal`
are optional and imputed by the pipeline. Every request is logged as one JSON line
with a request id (also returned as `X-Request-ID`).

```bash
uvicorn heartrisk.api:app --reload                      # local, no Docker
bash scripts/docker_smoke_test.sh                       # build + run + verify container
docker run -p 8000:8000 heartrisk-api:latest
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json
```

Image: two-stage build on `python:3.12-slim`, serving deps only
(`requirements-serve.txt`), non-root user, model baked in, built-in HEALTHCHECK.

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push / pull request to `main` (and manually):

```
lint ──▶ test ──▶ train ──▶ docker
```

| Job | Does | Fails when | Artifacts |
|---|---|---|---|
| lint | `ruff check` (errors annotated inline on the PR/commit) | any lint error | `lint-log` |
| test | 61 pytest tests + coverage | any test fails | `test-reports` (pytest log, JUnit XML, coverage XML + HTML) |
| train | download → full grid search → MLflow → sanity prediction | training error, or CV ROC-AUC < 0.85 | `model`, `training-outputs` (logs, figures, `mlruns.tar.gz`) |
| docker | builds the image **with the model trained in this run**, runs the smoke test | build fails / any endpoint check fails | `docker-smoke-log` |

Every job writes a summary (test counts, coverage, selected model + metrics, image
size) to the run page; logs are uploaded even when a job fails.

## Tests

```bash
ruff check src scripts tests
pytest --cov=heartrisk          # 61 tests, ~17 s, 98 % coverage
```

| File | Covers |
|---|---|
| `tests/test_data.py` | raw parsing (`?` → NaN), target binarisation, domain validation, zero-value handling, duplicates, schema errors, download (network mocked) |
| `tests/test_features.py` | engineered feature maths, column layout, imputation leaves no NaN, scaling, unseen categories, end-to-end pipeline |
| `tests/test_evaluation.py` | metric correctness, threshold behaviour, edge cases, plot generation |
| `tests/test_eda.py` | every EDA figure renders |
| `tests/test_api.py` | health, predict/batch contract, parity with offline pipeline, validation (422), optional fields, request-id, JSON logs, Prometheus metrics, 503 without model |
| `tests/test_train.py` | quick training run in a sandbox: all models trained, ROC-AUC > 0.8, best-on-CV selected, artefacts + MLflow runs written, **reproducible predictions across two runs** |

## One-command run (used for the lab submission)

```bash
python scripts/run_pipeline.py      # lint → data → eda → train → test → predict → docker
```

Writes per-step logs and `reports/run/run_summary.md` (host, OS, versions, commit,
pass/fail). See [LAB_RUNBOOK.md](LAB_RUNBOOK.md) for the full lab procedure.
