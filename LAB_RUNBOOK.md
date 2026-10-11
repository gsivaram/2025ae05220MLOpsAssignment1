# Lab Runbook — executing and submitting from the lab

Development happens on a personal machine; **every result that is submitted is
produced on the lab machine** by following this runbook. The pipeline runner
records the host name, OS, Python and package versions and git commit of each run
in `reports/run/run_summary.md` as proof of where it was executed.

---

## 0. Prerequisites (check once)

```bash
python3 --version      # 3.10 or newer (Windows: python --version)
git --version
```

## 1. Get the code

```bash
git clone https://github.com/gsivaram/2025ae05220MLOpsAssignment1.git
cd 2025ae05220MLOpsAssignment1
```

The repository is private: when prompted, username = your GitHub user, password =
a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens (classic),
scopes `repo` and `workflow`). Never type your GitHub password here.

## 2. Create the environment

**Lab VM (Rocky Linux 9 + conda):** the system `python3` is **3.9 — too old**. Build the
venv from conda's Python 3.12, then leave conda (the venv is self-contained):

```bash
conda activate base && python --version     # must be 3.12.x
python -m venv .venv                        # or: $(conda info --base)/bin/python -m venv .venv
conda deactivate
source .venv/bin/activate && python --version   # still 3.12.x, prompt shows (.venv)
pip install --upgrade pip
pip install -r requirements.txt && pip install -e .
```

Every new terminal: `cd ~/2025ae05220MLOpsAssignment1 && source .venv/bin/activate`.

Other machines:

| Linux / macOS | Windows (PowerShell) |
|---|---|
| `python3 -m venv .venv` | `python -m venv .venv` |
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| `pip install -e .` | `pip install -e .` |

Behind a proxy: add `--proxy http://host:port` to the `pip` commands.

## 3. Run the full pipeline (one command)

```bash
python scripts/run_pipeline.py
```

| Step | What it does | Time |
|---|---|---|
| lint | ruff static checks | <1 s |
| data | downloads UCI file (falls back to the committed copy if offline), writes cleaned CSV | ~2 s |
| eda | executes `notebooks/01_eda.ipynb`, regenerates figures 01–06 | ~4 s |
| train | 116-configuration grid search, MLflow logging, saves `models/` + figures 07–10 | ~20 s |
| test | 61 unit tests (data, features, training, API) with coverage | ~17 s |
| predict | scores `sample_request.json` | ~3 s |
| docker | builds `heartrisk-api` image, runs it, checks /health, /predict, 422 on bad input, /metrics, HEALTHCHECK | ~2 min first time, then ~10 s |

Expected: `PIPELINE PASSED`, selected model **logistic_regression**, CV ROC-AUC
**0.9075**, test ROC-AUC **0.9610** (identical on every machine — fixed seeds).

If a step fails, the run stops there; read `reports/run/<step>.log`, fix, then resume
with `python scripts/run_pipeline.py --from <step>`.

## 4. Take screenshots (save into `screenshots/`, overwriting the old files)

**Terminal evidence**

| File | What to capture |
|---|---|
| `lab_00_environment.png` | top of the runner output — Host / OS / Python / commit |
| `lab_01_pipeline_passed.png` | the step results and `PIPELINE PASSED` line |
| `tests_01_pytest_coverage.png` | `pytest --cov=heartrisk` — passed count + coverage table |

| `docker_01_build.png` | `docker build -t heartrisk-api:latest .` finishing, then `docker image ls heartrisk-api` |
| `docker_02_smoke_test.png` | `bash scripts/docker_smoke_test.sh` — the /health, /predict JSON and `DOCKER SMOKE TEST PASSED` |

**API in the browser** — `docker run -d --name heartrisk -p 8000:8000 heartrisk-api:latest`, then in Firefox:

| File | Where |
|---|---|
| `api_01_swagger.png` | http://127.0.0.1:8000/docs — endpoint list |
| `api_02_predict.png` | `/docs` → **POST /predict** → *Try it out* → *Execute* (example body is pre-filled) → response |
| `api_03_metrics.png` | http://127.0.0.1:8000/metrics |

Stop it afterwards: `docker rm -f heartrisk`

**MLflow UI** — start with `mlflow ui --backend-store-uri ./mlruns`, open http://127.0.0.1:5000

| File | Where |
|---|---|
| `mlflow_01_experiment_runs.png` | experiment *heart-disease-risk* → Runs table (add columns `cv_roc_auc_mean`, `test_roc_auc`, `test_recall`) |
| `mlflow_02_child_runs.png` | expand **+** on `logistic_regression`, sort by `cv_roc_auc_mean` |
| `mlflow_03_chart_view.png` | search `tags.run_level = "model"` → Enter → **Chart** view, search charts `roc_auc` |
| `mlflow_04_parallel_coords.png` | expand `random_forest`, select children → **Compare** → Parallel Coordinates (params `max_depth`, `min_samples_leaf`, `max_features`, `hr_reserve_feature`; metric `cv_roc_auc_mean`) |
| `mlflow_05_selected_run_overview.png` | open `logistic_regression` parent run → Overview (tags incl. `selected=true`, params, metrics) |
| `mlflow_06_artifacts_plots.png` | same run → Artifacts → `plots/confusion_matrix.png` |
| `mlflow_07_logged_model.png` | same run → Artifacts → `model/` (signature) |
| `mlflow_08_model_selection.png` | `model_selection` run → Artifacts → `plots/model_comparison.png` |

## 5. Commit the lab results

```bash
git add -A
git commit -m "Lab run: pipeline results and screenshots"
git push
```

This commits the lab-generated `reports/run/`, `models/`, `reports/figures/`, the executed
notebook and the screenshots. (`mlruns/` is intentionally not committed — it is
regenerated by the `train` step.)

---

*Phases 4–7 (Docker, CI, Kubernetes, monitoring) will extend this runbook.*
