"""Score one or more patients with the saved model pipeline.

Usage:
    python scripts/predict.py sample_request.json          # JSON object or list of objects
    python scripts/predict.py data/processed/heart_clean.csv --limit 5
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from heartrisk import config
from heartrisk.train import METADATA_FILE, MODEL_FILE


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    payload = json.loads(path.read_text())
    return pd.DataFrame(payload if isinstance(payload, list) else [payload])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="JSON (object or list) or CSV with the 13 clinical features")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    args = parser.parse_args()

    pipeline = joblib.load(MODEL_FILE)
    meta = json.loads(METADATA_FILE.read_text())
    threshold = meta["decision_threshold"]

    df = load_input(args.input).head(args.limit)
    proba = pipeline.predict_proba(df[config.FEATURES])[:, 1]
    out = pd.DataFrame({
        "prediction": (proba >= threshold).astype(int),
        "label": ["disease" if p >= threshold else "no disease" for p in proba],
        "probability_disease": proba.round(4),
    })
    if config.TARGET in df:
        out.insert(0, "actual", df[config.TARGET].to_numpy())

    print(f"model: {meta['model_name']} v{meta['model_version']}  (threshold {threshold})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
