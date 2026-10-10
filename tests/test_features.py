import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from heartrisk import config
from heartrisk.features import ENGINEERED_FEATURES, ClinicalFeatures, build_pipeline, build_preprocessor, split_xy


def test_hr_reserve_value(patient):
    out = ClinicalFeatures().fit_transform(pd.DataFrame([patient]))
    assert out["hr_reserve"].iloc[0] == pytest.approx(150 / (220 - 63))


def test_hr_reserve_disabled_keeps_column_layout(patient):
    out = ClinicalFeatures(enabled=False).fit_transform(pd.DataFrame([patient]))
    assert list(out.columns) == config.FEATURES + ENGINEERED_FEATURES
    assert out["hr_reserve"].iloc[0] == 0.0


def test_clinical_features_rejects_missing_inputs(patient):
    with pytest.raises(ValueError, match="thalach"):
        ClinicalFeatures().fit(pd.DataFrame([patient]).drop(columns="thalach"))


def test_clinical_features_ignores_extra_columns_and_order(patient):
    df = pd.DataFrame([{"patient_id": 7, **dict(reversed(list(patient.items())))}])
    out = ClinicalFeatures().fit_transform(df)
    assert list(out.columns) == config.FEATURES + ENGINEERED_FEATURES


def test_preprocessor_output_is_dense_finite_and_expected_width(clean_df):
    X, _ = split_xy(clean_df)
    Xt = build_preprocessor().fit_transform(ClinicalFeatures().fit_transform(X))
    # 7 numeric + one-hot(cp:4, restecg:3, slope:3, thal:3) + 3 binary
    assert Xt.shape == (len(X), 7 + 13 + 3)
    assert np.isfinite(Xt).all(), "imputation must remove every NaN"


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor()
    Xt = pd.DataFrame(pre.fit_transform(ClinicalFeatures().fit_transform(X)), columns=pre.get_feature_names_out())
    for col in config.NUMERIC_FEATURES:
        assert Xt[col].mean() == pytest.approx(0, abs=1e-9)
        assert Xt[col].std(ddof=0) == pytest.approx(1, abs=1e-6)


def test_pipeline_handles_missing_and_unseen_values(clean_df, patient):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(DummyClassifier(strategy="prior")).fit(X, y)
    odd = pd.DataFrame([{**patient, "ca": np.nan, "thal": np.nan}, {**patient, "cp": 99}])
    proba = pipe.predict_proba(odd)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_split_xy(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.FEATURES
    assert y.name == config.TARGET
