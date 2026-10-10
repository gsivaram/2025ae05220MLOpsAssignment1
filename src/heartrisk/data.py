"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heartrisk import config

log = logging.getLogger(__name__)


def download_raw(dest: Path = config.RAW_FILE, url: str = config.UCI_URL, force: bool = False) -> Path:
    """Download the raw Cleveland file from UCI. Skips if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("Raw file already present at %s", dest)
        return dest
    log.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = resp.read()
    except OSError as exc:
        # Offline / firewalled machines (e.g. a lab) fall back to the committed copy.
        if dest.exists():
            log.warning("Download failed (%s); using existing file %s", exc, dest)
            return dest
        raise
    dest.write_bytes(payload)
    log.info("Saved %d bytes (sha256=%s)", len(payload), hashlib.sha256(payload).hexdigest()[:16])
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the header-less raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, validated frame with a binary target.

    Missing values are *kept* as NaN on purpose: imputation is learned inside
    the model pipeline (fit on training folds only) to avoid leakage.
    """
    df = raw.copy()

    missing_cols = set(config.RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing columns: {sorted(missing_cols)}")

    df = df.apply(pd.to_numeric, errors="coerce")

    # Codes outside the documented domain become missing rather than silently
    # creating a new category.
    for col, valid in config.VALID_CODES.items():
        df.loc[~df[col].isin(valid) & df[col].notna(), col] = np.nan

    # Physiologically impossible readings are treated as missing.
    df.loc[df["trestbps"] <= 0, "trestbps"] = np.nan
    df.loc[df["chol"] <= 0, "chol"] = np.nan

    # num is 0 (no disease) .. 4 (severe). The task is presence vs absence.
    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        log.info("Dropped %d duplicate rows", before - len(df))

    return df[config.FEATURES + [config.TARGET]]


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Count of missing values per column (only columns with any)."""
    counts = df.isna().sum()
    return counts[counts > 0]


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def build_clean_dataset(force_download: bool = False) -> pd.DataFrame:
    """End-to-end: download (if needed), clean, and write the processed CSV."""
    download_raw(force=force_download)
    df = clean(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Wrote %s (%d rows, %d cols)", config.CLEAN_FILE, *df.shape)
    return df
