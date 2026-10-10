# Heart Disease Risk — End-to-End MLOps

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

## Tests

```bash
ruff check src scripts tests
pytest --cov=heartrisk          # 43 tests, ~17 s, 98 % coverage
```

| File | Covers |
|---|---|
| `tests/test_data.py` | raw parsing (`?` → NaN), target binarisation, domain validation, zero-value handling, duplicates, schema errors, download (network mocked) |
| `tests/test_features.py` | engineered feature maths, column layout, imputation leaves no NaN, scaling, unseen categories, end-to-end pipeline |
| `tests/test_evaluation.py` | metric correctness, threshold behaviour, edge cases, plot generation |
| `tests/test_eda.py` | every EDA figure renders |
| `tests/test_train.py` | quick training run in a sandbox: all models trained, ROC-AUC > 0.8, best-on-CV selected, artefacts + MLflow runs written, **reproducible predictions across two runs** |

## One-command run (used for the lab submission)

```bash
python scripts/run_pipeline.py      # lint → data → eda → train → test → predict
```

Writes per-step logs and `reports/run/run_summary.md` (host, OS, versions, commit,
pass/fail). See [LAB_RUNBOOK.md](LAB_RUNBOOK.md) for the full lab procedure.
