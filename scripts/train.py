"""Train, tune and compare the candidate models, logging everything to MLflow.

Usage:
    python scripts/train.py            # full grid search (~200 configurations)
    python scripts/train.py --quick    # small grids, used by CI
    mlflow ui --backend-store-uri ./mlruns   # browse the runs at http://127.0.0.1:5000
"""

import argparse
import logging
import warnings

from heartrisk.train import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="use reduced grids (fast smoke run for CI)")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (default: ./mlruns)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    summary = run(quick=args.quick, tracking_uri=args.tracking_uri)
    cols = [c for c in summary.columns if c.endswith("_mean") or c.startswith("test_")]
    print("\nModel summary (CV = 5-fold on training split, test = held-out 20%):")
    print(summary[[c for c in cols if "std" not in c]].round(3).T.to_string())


if __name__ == "__main__":
    main()
