"""Download the UCI Heart Disease dataset and write the cleaned CSV.

Usage:
    python scripts/download_data.py            # download if missing, then clean
    python scripts/download_data.py --force    # re-download from UCI
"""

import argparse
import logging

from heartrisk.data import build_clean_dataset, missing_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the raw file exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = build_clean_dataset(force_download=args.force)

    print(f"\nClean dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Missing values (imputed later inside the model pipeline):")
    print(missing_report(df).to_string() or "  none")
    print("\nClass balance:")
    print(df["target"].value_counts().rename({0: "no disease", 1: "disease"}).to_string())


if __name__ == "__main__":
    main()
