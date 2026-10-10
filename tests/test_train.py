"""Training smoke test: the full quick pipeline runs end-to-end in an isolated sandbox."""

import json

import joblib
import mlflow
import numpy as np
import pytest

from heartrisk import config
from heartrisk import train as train_mod


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    sandbox = tmp_path_factory.mktemp("train")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "FIGURES_DIR", sandbox / "figures")
    mp.setattr(config, "MODELS_DIR", sandbox / "models")
    mp.setattr(train_mod, "MODEL_FILE", sandbox / "models" / "model.joblib")
    mp.setattr(train_mod, "METADATA_FILE", sandbox / "models" / "metadata.json")
    uri = (sandbox / "mlruns").as_uri()
    summary = train_mod.run(quick=True, tracking_uri=uri)
    yield {"sandbox": sandbox, "summary": summary, "uri": uri}
    mp.undo()


def test_every_candidate_is_trained(trained):
    assert set(trained["summary"].index) == {c.name for c in train_mod.candidates()}


def test_models_beat_a_useful_baseline(trained):
    # Prevalence is 46%, so a no-skill model has ROC-AUC 0.5; anything below 0.8 signals a bug.
    assert (trained["summary"]["cv_roc_auc_mean"] > 0.8).all()


def test_artifacts_written(trained):
    models = trained["sandbox"] / "models"
    meta = json.loads((models / "metadata.json").read_text())
    assert meta["model_name"] in trained["summary"].index
    assert meta["features"] == config.FEATURES
    assert {"cv_roc_auc_mean", "test_roc_auc", "test_recall"} <= meta["metrics"].keys()
    assert (trained["sandbox"] / "figures" / "07_model_comparison.png").exists()


def test_selected_model_is_best_on_cv(trained):
    meta = json.loads((trained["sandbox"] / "models" / "metadata.json").read_text())
    assert meta["model_name"] == trained["summary"]["cv_roc_auc_mean"].idxmax()


def test_saved_pipeline_predicts_valid_probabilities(trained, clean_df):
    pipe = joblib.load(trained["sandbox"] / "models" / "model.joblib")
    proba = pipe.predict_proba(clean_df[config.FEATURES])[:, 1]
    assert proba.shape == (len(clean_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_training_is_reproducible(trained, clean_df, tmp_path, monkeypatch):
    """Same data + seed ⇒ identical predictions from a second, independent run."""
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_mod, "MODEL_FILE", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(train_mod, "METADATA_FILE", tmp_path / "models" / "metadata.json")
    train_mod.run(quick=True, tracking_uri=(tmp_path / "mlruns").as_uri())
    a = joblib.load(trained["sandbox"] / "models" / "model.joblib").predict_proba(clean_df[config.FEATURES])
    b = joblib.load(tmp_path / "models" / "model.joblib").predict_proba(clean_df[config.FEATURES])
    np.testing.assert_allclose(a, b)


def test_mlflow_runs_logged(trained):
    mlflow.set_tracking_uri(trained["uri"])
    runs = mlflow.search_runs(experiment_names=[train_mod.EXPERIMENT_NAME])
    levels = runs["tags.run_level"].value_counts().to_dict()
    assert levels["model"] == 3 and levels["selection"] == 1 and levels["grid_candidate"] >= 3
    assert (runs["tags.selected"] == "true").sum() == 1
