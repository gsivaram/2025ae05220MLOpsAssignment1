"""Model training, tuning and experiment tracking.

For every candidate model:
  * a GridSearchCV over (model hyper-parameters x engineered-feature on/off),
    scored with stratified 5-fold CV on the training split;
  * one MLflow parent run with the best params, CV mean/std metrics, held-out
    test metrics, diagnostic plots and the fitted pipeline;
  * one nested MLflow child run per grid candidate, so every configuration
    tried is visible and comparable in the MLflow UI.

The winner is chosen on CV ROC-AUC only (the test split is never used for
selection) and written to models/ as a joblib pipeline + metadata JSON.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heartrisk import __version__, config
from heartrisk.data import build_clean_dataset, load_clean
from heartrisk.evaluation import (
    CV_SCORING,
    compute_metrics,
    permutation_importances,
    plot_confusion,
    plot_importance,
    plot_model_comparison,
    plot_pr,
    plot_roc,
)
from heartrisk.features import build_pipeline, split_xy

log = logging.getLogger(__name__)

EXPERIMENT_NAME = "heart-disease-risk"
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"
MODEL_FILE = config.MODELS_DIR / "model.joblib"
METADATA_FILE = config.MODELS_DIR / "metadata.json"
MODEL_REQUIREMENTS = [
    f"scikit-learn=={sklearn.__version__}",
    f"pandas=={pd.__version__}",
    f"numpy=={np.__version__}",
    f"joblib=={joblib.__version__}",
]


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict[str, list]
    quick_grid: dict[str, list]


def candidates() -> list[Candidate]:
    rs = config.RANDOM_STATE
    return [
        Candidate(
            "logistic_regression",
            LogisticRegression(solver="liblinear", max_iter=5000, random_state=rs),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
                "features__enabled": [True, False],
            },
            {"model__C": [0.1, 1.0], "features__enabled": [True, False]},
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(n_jobs=1, random_state=rs),
            {
                "model__n_estimators": [300],
                "model__max_depth": [None, 4, 6],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
                "features__enabled": [True, False],
            },
            {"model__n_estimators": [100], "model__max_depth": [4, None]},
        ),
        Candidate(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [100, 250],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.7, 1.0],
                "features__enabled": [True, False],
            },
            {"model__n_estimators": [100], "model__max_depth": [2]},
        ),
    ]


def _param_name(key: str) -> str:
    return "hr_reserve_feature" if key == "features__enabled" else key.removeprefix("model__")


def _readable_params(params: dict) -> dict:
    return {_param_name(k): ("None" if v is None else v) for k, v in params.items()}


def _cv_summary(cv_results: dict, idx: int) -> dict[str, float]:
    out = {}
    for m in CV_SCORING:
        out[f"cv_{m}_mean"] = float(cv_results[f"mean_test_{m}"][idx])
        out[f"cv_{m}_std"] = float(cv_results[f"std_test_{m}"][idx])
    return out


def _log_figure(fig: plt.Figure, name: str, report_name: str | None = None) -> None:
    """Log to the active MLflow run; optionally also save under reports/figures for the write-up."""
    mlflow.log_figure(fig, f"plots/{name}.png")
    if report_name:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / f"{report_name}.png")
    plt.close(fig)


def _log_child_runs(name: str, search: GridSearchCV) -> None:
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-{i:03d}", nested=True):
            mlflow.set_tags({"model_family": name, "run_level": "grid_candidate",
                             "rank": int(res[f"rank_test_{SELECTION_METRIC}"][i])})
            mlflow.log_params(_readable_params(params))
            mlflow.log_metrics(_cv_summary(res, i))
            mlflow.log_metric("fit_time_mean_s", float(res["mean_fit_time"][i]))


def train_candidate(cand: Candidate, X_train, y_train, X_test, y_test, quick: bool) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    grid = cand.quick_grid if quick else cand.grid
    search = GridSearchCV(build_pipeline(cand.estimator), grid, scoring=CV_SCORING,
                          refit=SELECTION_METRIC, cv=cv, n_jobs=-1)

    with mlflow.start_run(run_name=cand.name) as run:
        mlflow.set_tags({"model_family": cand.name, "run_level": "model", "quick_mode": quick})
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        log.info("%s: %d candidates in %.1fs, best CV %s=%.4f", cand.name,
                 len(search.cv_results_["params"]), elapsed, SELECTION_METRIC, search.best_score_)

        _log_child_runs(cand.name, search)

        best = search.best_estimator_
        proba = best.predict_proba(X_test)[:, 1]
        cv_metrics = _cv_summary(search.cv_results_, search.best_index_)
        test_metrics = {f"test_{k}": float(v) for k, v in compute_metrics(y_test, proba).items()}

        mlflow.log_params({"cv_folds": CV_FOLDS, "test_size": TEST_SIZE, "n_grid_candidates":
                           len(search.cv_results_["params"]), **_readable_params(search.best_params_)})
        mlflow.log_metrics({**cv_metrics, **test_metrics, "search_time_s": elapsed})

        _log_figure(plot_roc({cand.name: (y_test, proba)}), "roc_curve")
        _log_figure(plot_pr({cand.name: (y_test, proba)}), "pr_curve")
        _log_figure(plot_confusion(y_test, proba), "confusion_matrix")
        imp = permutation_importances(best, X_test, y_test)
        _log_figure(plot_importance(imp), "permutation_importance")

        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(Path(tmp, "cv_results.csv"), index=False)
            imp.to_csv(Path(tmp, "permutation_importance.csv"), index=False)
            mlflow.log_artifacts(tmp, artifact_path="tables")

        signature = infer_signature(X_test, best.predict_proba(X_test))
        # Bundle the heartrisk package (custom transformer) and pin runtime deps so the
        # logged model loads in a clean environment without installing this repo.
        mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                 input_example=X_test.head(3),
                                 code_paths=[str(Path(__file__).resolve().parent)],
                                 pip_requirements=MODEL_REQUIREMENTS)

        return {"name": cand.name, "run_id": run.info.run_id, "pipeline": best, "proba": proba,
                "best_params": _readable_params(search.best_params_), **cv_metrics, **test_metrics}


def save_model(result: dict, n_train: int, n_test: int) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], MODEL_FILE)
    metadata = {
        "model_name": result["name"],
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": result["run_id"],
        "selection_metric": f"cv_{SELECTION_METRIC}_mean",
        "decision_threshold": 0.5,
        "features": config.FEATURES,
        "best_params": result["best_params"],
        "metrics": {k: round(v, 4) for k, v in result.items() if k.startswith(("cv_", "test_"))},
        "n_train": n_train,
        "n_test": n_test,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))
    log.info("Saved %s and %s", MODEL_FILE, METADATA_FILE)


def run(quick: bool = False, tracking_uri: str | None = None) -> pd.DataFrame:
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or (config.PROJECT_ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow tracking URI: %s", tracking_uri)

    if not config.CLEAN_FILE.exists():
        build_clean_dataset()
    X, y = split_xy(load_clean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    log.info("Train %d rows, test %d rows", len(X_train), len(X_test))

    results = [train_candidate(c, X_train, y_train, X_test, y_test, quick) for c in candidates()]

    summary = (pd.DataFrame([{k: v for k, v in r.items() if k not in ("pipeline", "proba", "best_params")}
                             for r in results]).set_index("name"))
    winner_name = summary[f"cv_{SELECTION_METRIC}_mean"].idxmax()
    winner = next(r for r in results if r["name"] == winner_name)

    with mlflow.start_run(run_name="model_selection"):
        mlflow.set_tags({"run_level": "selection", "selected_model": winner_name,
                         "selected_run_id": winner["run_id"]})
        mlflow.log_param("selection_metric", f"cv_{SELECTION_METRIC}_mean")
        mlflow.log_metrics({f"selected_{k}": v for k, v in summary.loc[winner_name].items()
                            if isinstance(v, (int, float, np.floating))})
        _log_figure(plot_model_comparison(summary), "model_comparison", "07_model_comparison")
        _log_figure(plot_roc({r["name"]: (y_test, r["proba"]) for r in results},
                             title="ROC curves — all models, test set"), "roc_all_models", "08_roc_all_models")
        _log_figure(plot_confusion(y_test, winner["proba"], title=f"Confusion matrix — {winner_name}"),
                    "selected_confusion_matrix", "09_selected_confusion_matrix")
        _log_figure(plot_importance(permutation_importances(winner["pipeline"], X_test, y_test),
                                    title=f"Permutation importance — {winner_name}"),
                    "selected_permutation_importance", "10_selected_permutation_importance")
        with tempfile.TemporaryDirectory() as tmp:
            summary.round(4).to_csv(Path(tmp, "model_summary.csv"))
            mlflow.log_artifacts(tmp, artifact_path="tables")

    mlflow.tracking.MlflowClient().set_tag(winner["run_id"], "selected", "true")

    save_model(winner, len(X_train), len(X_test))
    log.info("Selected %s (CV ROC-AUC %.4f, test ROC-AUC %.4f)", winner_name,
             winner[f"cv_{SELECTION_METRIC}_mean"], winner[f"test_{SELECTION_METRIC}"])
    return summary
