# Heart Disease Risk — End-to-End MLOps

Binary classifier predicting presence of heart disease from 13 clinical features
(UCI Heart Disease, Cleveland subset), packaged as a monitored, containerised API.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python scripts/download_data.py          # data/raw -> data/processed/heart_clean.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Repository layout

| Path | Contents |
|---|---|
| `src/heartrisk/` | Library code: config, data cleaning, EDA plots |
| `scripts/` | CLI entry points (download, train, …) |
| `notebooks/` | Narrative notebooks (EDA, …) |
| `data/raw`, `data/processed` | Raw UCI file and cleaned CSV |
| `reports/figures/` | Generated plots |
| `tests/` | Unit tests |
| `screenshots/` | Evidence for the report |

## Dataset

UCI Machine Learning Repository, *Heart Disease* (id 45),
`processed.cleveland.data` — 303 rows. Downloaded automatically by
`scripts/download_data.py`; target `num` (0–4) is binarised to presence (1) / absence (0).
