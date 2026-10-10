"""Feature engineering and the preprocessing pipeline.

Everything here is a scikit-learn transformer so that the exact same steps are
fit during training and replayed at inference time from one pickled Pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk import config

ENGINEERED_FEATURES = ["hr_reserve"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds domain-derived features to the raw clinical columns.

    hr_reserve: peak heart rate achieved as a fraction of the age-predicted
    maximum (220 - age). EDA showed diseased patients sit well below that line
    at every age, which a linear model cannot express from age and thalach alone.

    ``enabled`` is a hyper-parameter so the grid search can test whether the
    feature actually helps; when disabled the column is still emitted (as NaN
    → imputed constant) so the downstream column layout never changes.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(config.FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing required features: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[config.FEATURES].copy()
        if self.enabled:
            predicted_max = (220 - X["age"]).where(lambda s: s > 0)
            X["hr_reserve"] = X["thalach"] / predicted_max
        else:
            X["hr_reserve"] = 0.0
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.FEATURES + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Impute → scale numerics; impute → one-hot nominal codes; binaries pass through."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model) -> Pipeline:
    """Full inference pipeline: raw clinical columns in, class probabilities out."""
    return Pipeline([
        ("features", ClinicalFeatures()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES], df[config.TARGET]
