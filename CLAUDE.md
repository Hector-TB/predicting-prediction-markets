# CLAUDE.md — Predicting Prediction Markets

This file is read by Claude Code at the start of every session. Keep it up to date.

---

## What this project is

ML models trained on historical [Polymarket](https://polymarket.com) binary prediction markets to forecast YES/NO outcomes. The core question: can ML beat the market's own implied probability?

Originally a course project (NYU DS-GA 1003, team of 3). Now being productionized into a full-stack application.

**Long-term target architecture:** data pipeline → PostgreSQL database → FastAPI backend → React frontend → model serving layer.

---

## Current state (as of 2026-09-19)

- Phase: cleanup + foundation (steps 1–4 of productionization plan)
- Pipeline scripts work and have been run; output parquet files are in `data/`
- Models trained: logistic regression, XGBoost, SVM (with and without trends), random forest (notebook only)
- Incomplete: random_forest has notebook only (no train.py), lightgbm is an empty placeholder
- Database: Supabase Postgres schema created; `db/load_parquet.py` loads metadata + model registry
- No API, no frontend yet
- See `docs/decisions/` for Architecture Decision Records

---

## Repo structure

```
predicting-prediction-markets/
├── data/                          # parquet datasets (large — do not modify manually)
├── data_collection_pipeline/      # ETL scripts — run in numbered order
├── models/
│   ├── common/                    # shared evaluation + preprocessing utilities
│   ├── gradient_boosting/         # XGBoost: train.py + notebook + predictions/
│   ├── logistic_regression/       # train.py + notebook + predictions/
│   ├── random_forest/             # notebook only (train.py not yet written)
│   ├── random_forest_trends/      # notebook only (train.py not yet written)
│   └── svm/                       # svm.py + svm_evaluate.py + notebook + predictions/
├── analysis/                      # analysis.ipynb, trading_simulation.ipynb, explore_data.py
├── notebooks/                     # scratch EDA
├── plots/                         # generated PNG output (gitignored eventually)
├── docs/
│   └── decisions/                 # Architecture Decision Records (ADRs)
└── pyproject.toml
```

---

## Data pipeline — run in order

```bash
# Step 1: fetch resolved markets from Gamma API
python data_collection_pipeline/fetch_markets.py

# Step 2: build 12-hour snapshot dataset
python data_collection_pipeline/build_snapshots.py

# Step 3: data quality fixes (dedupe, clip, NaN fill)
python data_collection_pipeline/fix_dataset.py

# Step 4: LLM category assignment (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... python data_collection_pipeline/categorize_markets.py

# Step 5: Google Trends enrichment
python data_collection_pipeline/fetch_category_trends.py
python data_collection_pipeline/build_trend_features.py
python data_collection_pipeline/merge_trends.py

# Step 6: remove post-resolution snapshots (data leakage fix)
python data_collection_pipeline/fix_leakage.py
```

All scripts resolve paths relative to `__file__`, so they work from any working directory.

The canonical dataset for modelling is `data/polymarket_ml_dataset_clean.parquet` (base features) or `data/polymarket_ml_dataset_with_trends_clean.parquet` (with Google Trends).

**Data files live on S3** (ADR-012). On a new machine, pull them before training:
```bash
python data/sync.py pull     # download all parquet files from S3
python data/sync.py push     # upload after re-running the pipeline
python data/sync.py status   # compare local vs S3
```

---

## Training models

```bash
python models/logistic_regression/train.py
python models/gradient_boosting/train.py
python models/svm/svm.py && python models/svm/svm_evaluate.py
```

Shared evaluation utilities are in `models/common/evaluation.py`. Import from there — do not duplicate.

---

## Key conventions

**Path resolution:** always use `Path(__file__).resolve().parent` — never hardcode relative paths like `"data/..."`.

**Metrics:** primary metrics are AUC-ROC and log-loss. The market price baseline is AUC-ROC = 0.964 / log-loss = 0.171 — beating this is the bar. Do not report raw accuracy as a primary metric (class imbalance: 78% NO).

**Class imbalance:** all models use `class_weight='balanced'` or equivalent (`scale_pos_weight` for XGBoost). Always.

**Train/test split:** market-level temporal split — oldest 80% train, newest 20% test. Never mix snapshots from the same market across splits. See ADR-001.

**Data files:** parquet files in `data/` are large and should eventually move out of git (to S3 or similar). Do not commit new large files. The `_clean` suffix means leakage-fixed.

**No bare `print()` for logging in new code** — use Python's `logging` module. Existing scripts use print; leave them as-is until refactored.

---

## Environment

Python 3.13+. Install with:
```bash
pip install -e ".[dev]"
```

`ANTHROPIC_API_KEY` is only required for `categorize_markets.py`.

---

## What NOT to do

- Do not modify `data/polymarket_ml_dataset.parquet` or `data/polymarket_ml_dataset_clean.parquet` directly — these are derived from the pipeline and must be reproducible
- Do not add new model-specific evaluation functions — put them in `models/common/evaluation.py`
- Do not amend commits that have been pushed
- Do not change the train/test split logic without a new ADR

---

## Database

**Split: parquet for training, Postgres for operations (ADR-011).**

- Training scripts read parquet directly — never query the database
- The DB holds: `markets` (metadata), `trends`, `model_runs` (with metrics JSONB)
- The DB will hold `snapshots` + `predictions` for **live markets only** (not backfilled with 1.4M historical rows)
- To populate the DB: `python db/load_parquet.py` (completes in seconds)
- Pending migration before first run: `db/migrations/004_add_model_run_metrics.sql`

---

## Architecture decisions

All significant decisions are documented in `docs/decisions/`. Read the relevant ADR before changing any of the following:
- Train/test split methodology → ADR-001
- Snapshot window parameters → ADR-002
- Rolling window selection → ADR-003
- Outcome threshold → ADR-004
- Leakage definition and fix → ADR-005
- SVM subsampling strategy → ADR-006
- Class imbalance handling → ADR-007
- Market filters → ADR-008
- Google Trends enrichment approach → ADR-009
- Database design → ADR-010
- Offline/online split (parquet vs DB) → ADR-011
- S3 + DuckDB data lake → ADR-012
