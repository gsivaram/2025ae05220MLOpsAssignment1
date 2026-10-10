"""Shared fixtures. Tests never touch the network or the real models/ and mlruns/ folders."""

import pandas as pd
import pytest

from heartrisk import config
from heartrisk.data import clean, load_raw


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The committed raw UCI file (303 rows)."""
    return load_raw(config.RAW_FILE)


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture
def patient() -> dict:
    """One valid patient record, as the API / predict script receive it."""
    return {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
            "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture
def raw_row(patient) -> dict:
    """The same patient in raw-file form (with the 0-4 `num` target)."""
    return {**patient, "num": 2}
