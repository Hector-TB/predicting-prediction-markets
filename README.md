# Predicting Prediction Markets

ML models trained on historical [Polymarket](https://polymarket.com) binary prediction market data to forecast YES/NO outcomes. The core question: **can ML beat the market's own implied probability?**

Originally a course project (NYU DS-GA 1003, team of 3). Now being productionized into a full-stack application.

**Team:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri

---

## Research Questions

| | Question |
|---|---|
| **RQ1** | Can ML models outperform prediction market probabilities in forecasting outcomes? |
| **RQ2** | Do engineered features beyond market price enhance predictive performance? |
| **RQ3** | At which stage of a market's lifecycle do external signals provide the most value? |

---

## Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy environment config and fill in secrets
cp .env.example .env

# 3. Pull data files from S3
python data/sync.py pull
```

`.env` requires:
- `DATABASE_URL` — Supabase Postgres connection string
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_BUCKET` — S3 data lake
- `ANTHROPIC_API_KEY` — only needed to re-run `categorize_markets.py`

See `.env.example` for the full list.

---

## Repository Structure

```
predicting-prediction-markets/
├── data/                              # parquet/CSV data files (gitignored — live on S3)
│   └── sync.py                        # push/pull data files to/from S3
├── data_collection_pipeline/          # ETL scripts — run in numbered order
│   ├── fetch_markets.py               # Step 1: fetch resolved markets from Gamma API
│   ├── build_snapshots.py             # Step 2: 12-hour snapshot dataset + rolling features
│   ├── fix_dataset.py                 # Step 3: dedupe, clip prices, fill NaN features
│   ├── categorize_markets.py          # Step 4: LLM category assignment (Claude API)
│   ├── fetch_category_trends.py       # Step 5a: pull Google Trends data
│   ├── build_trend_features.py        # Step 5b: compute trend features
│   ├── merge_trends.py                # Step 5c: merge trends into snapshot dataset
│   └── fix_leakage.py                 # Step 6: remove post-resolution snapshots
├── models/
│   ├── common/                        # shared evaluation utilities (evaluation.py)
│   ├── logistic_regression/           # train.py + notebook + predictions/
│   ├── gradient_boosting/             # XGBoost: train.py + notebook + predictions/
│   ├── random_forest/                 # notebook only (train.py not yet written)
│   ├── random_forest_trends/          # notebook only (train.py not yet written)
│   ├── svm/                           # svm.py + svm_evaluate.py + notebook
│   └── lightgbm/                      # placeholder (not yet implemented)
├── analysis/
│   ├── analysis.ipynb                 # model comparison, lift curves, calibration
│   ├── trading_simulation.ipynb       # simulated trading strategy from predictions
│   └── explore_data.py                # data validation + diagnostic plots → plots/
├── db/
│   ├── load_parquet.py                # migrate metadata + model registry to Supabase
│   └── migrations/                    # SQL migration files (apply via Supabase MCP)
├── docs/
│   └── decisions/                     # Architecture Decision Records (ADR-001–012)
├── plots/                             # generated PNGs (gitignored)
├── .env.example
├── pyproject.toml
└── CLAUDE.md                          # persistent context for Claude Code sessions
```

---

## Data Pipeline

Run steps in order. All scripts resolve paths relative to `__file__`.

```bash
# Step 1: fetch resolved markets from Gamma API
python data_collection_pipeline/fetch_markets.py

# Step 2: build 12-hour snapshot dataset + rolling features
python data_collection_pipeline/build_snapshots.py

# Step 3: data quality fixes (dedupe, clip, NaN fill)
python data_collection_pipeline/fix_dataset.py

# Step 4: LLM category assignment (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... python data_collection_pipeline/categorize_markets.py

# Step 5: Google Trends enrichment
python data_collection_pipeline/fetch_category_trends.py
python data_collection_pipeline/build_trend_features.py
python data_collection_pipeline/merge_trends.py

# Step 6: remove post-resolution snapshots (data leakage fix — see ADR-005)
python data_collection_pipeline/fix_leakage.py
```

After re-running the pipeline, push updated files to S3:
```bash
python data/sync.py push
```

---

## Training Models

```bash
python models/logistic_regression/train.py
python models/gradient_boosting/train.py
python models/svm/svm.py && python models/svm/svm_evaluate.py
```

Shared evaluation utilities (AUC-ROC, PR-AUC, log-loss, Brier, calibration) are in `models/common/evaluation.py`.

---

## Results (test set, 288,490 snapshots across 4,174 markets)

**Baseline — market price alone:** AUC-ROC = **0.964**, log-loss = **0.171**

The market price is a near-perfect predictor. Beating it is the bar for RQ1.

| Model | AUC-ROC | Log-loss | Notes |
|---|---|---|---|
| Market price (baseline) | **0.964** | **0.171** | `price_at_snapshot` as the predictor |
| XGBoost + trends (calibrated) | 0.9034 | 0.3034 | Best ML model |
| XGBoost + trends | 0.9023 | 0.3070 | |
| Random forest + trends (calibrated) | 0.9019 | 0.3042 | |
| XGBoost base | 0.9000 | 0.3108 | |
| Random forest (calibrated) | 0.9018 | 0.3042 | |
| Random forest (full) | 0.9021 | 0.3635 | Uncalibrated RF overestimates |
| Logistic regression base | 0.8874 | 0.3451 | |
| Logistic regression + trends | 0.8868 | 0.3410 | |
| Random forest (price-only) | 0.8891 | 0.4194 | |

**Key finding:** No model beats the market baseline on either metric. The market's implied probability is already near-optimal. This is consistent with the efficient markets hypothesis — Polymarket's price aggregates information that ML features cannot improve upon in aggregate. The value of ML is likely in identifying specific market conditions or time windows where the model has an edge, not overall accuracy improvement.

Google Trends and calibration both help at the margin; XGBoost is the strongest architecture.

---

## Dataset

### Canonical files (on S3 — fetch with `python data/sync.py pull`)

| File | Rows | Description |
|---|---|---|
| `polymarket_ml_dataset_clean.parquet` | 1,448,142 | Base features — use for training |
| `polymarket_ml_dataset_with_trends_clean.parquet` | 1,448,142 | Base + Google Trends features |
| `category_trends_features.parquet` | ~1,500 | Weekly trend aggregates by category |
| `polymarket_markets_meta.csv` | 24,092 | Market metadata (regenerate with `fetch_markets.py`) |

### Markets (`polymarket_markets_meta.csv`)

24,092 resolved binary Polymarket markets. Filters: volume ≥ $1,000, duration ≥ 30 days.

| Column | Description |
|---|---|
| `market_id` | Polymarket condition ID |
| `clob_token_id` | YES token ID for CLOB API |
| `question` | Market question text |
| `category` | LLM-assigned category (9 classes — see below) |
| `start_date` | Market creation date |
| `end_date` | Scheduled close date |
| `duration_days` | `end_date − start_date` in days |
| `total_volume` | Lifetime trading volume (USD) |
| `yes_final_price` | Final settlement price of YES token |
| `outcome` | Ground truth: 1 = YES, 0 = NO |
| `split` | `"train"` or `"test"` (market-level temporal 80/20) |

### Snapshots (`polymarket_ml_dataset_clean.parquet`)

One row per (market × 12-hour snapshot). 1,448,142 rows, 27 columns.

Feature groups:
- **Time:** `days_before_close`, `pct_lifetime_elapsed`, `duration_days`
- **Price:** `price_at_snapshot`, `price_deviation_from_half`
- **Rolling 7d:** `price_mean_7d`, `price_volatility_7d`, `price_min_7d`, `price_max_7d`, `price_change_7d`, `price_range_7d`, `price_trend_7d`
- **Rolling 14d:** same set with `_14d` suffix
- **Volume:** `total_volume`, `log_volume`
- **Target:** `outcome`

### Train / Test Split

Market-level temporal split — oldest 80% of markets form the train set. No snapshot from a test market appears in training. See ADR-001.

| Split | Markets | Snapshots | YES rate |
|---|---|---|---|
| Train | 16,774 | 1,159,652 | 22.4% |
| Test | 4,174 | 288,490 | 21.0% |
| **Total** | **20,948** | **1,448,142** | **22.1%** |

3,144 markets (13%) are absent from the snapshot dataset: insufficient price observations or snapshot window collapses to zero (market resolved early).

### Class Imbalance

80.3% of markets resolve NO. A trivial always-NO classifier achieves ~78% accuracy — making raw accuracy misleading. **Primary metrics: AUC-ROC and log-loss.** All models use `class_weight='balanced'` or equivalent. See ADR-007.

### Google Trends Enrichment

`polymarket_ml_dataset_with_trends_clean.parquet` adds 5 columns per snapshot, joining on `(category, week_start)`:

| Column | Description |
|---|---|
| `trend_value` | Weekly Google Trends interest (0–100) for the market's category |
| `trend_ma4` | 4-week moving average |
| `trend_change_4w` | Change vs 4 weeks prior |
| `trend_spike` | 1 if `trend_value > 1.5 × trend_ma4` |
| `has_trend_data` | 1 if trend data exists for this category |

Coverage: 99.6% of snapshots. See ADR-009.

### Categories (LLM-assigned via Claude API)

| Category | Markets | % |
|---|---|---|
| `sports` | 7,081 | 29.4% |
| `politics_us` | 3,751 | 15.6% |
| `entertainment` | 3,553 | 14.7% |
| `finance` | 2,846 | 11.8% |
| `crypto` | 2,367 | 9.8% |
| `politics_global` | 1,900 | 7.9% |
| `geopolitics` | 1,292 | 5.4% |
| `science_tech` | 1,149 | 4.8% |
| `other` | 152 | 0.6% |

---

## Database

Supabase Postgres (operational layer — not used for training). Schema: `markets`, `snapshots`, `trends`, `model_runs`, `predictions`. Parquet files are the training data store; the DB serves live market scoring and the future frontend.

```bash
# Populate the DB (runs in seconds — metadata + model registry only)
python db/load_parquet.py
```

Pending migration (apply once DB is healthy): `db/migrations/004_add_model_run_metrics.sql`

See ADR-010 (database design) and ADR-011 (offline/online split).

---

## Architecture Decision Records

All significant decisions are documented in `docs/decisions/`. These are the audit trail for every methodological choice.

| ADR | Decision |
|---|---|
| [ADR-001](docs/decisions/001-temporal-train-test-split.md) | Temporal, market-level 80/20 train/test split |
| [ADR-002](docs/decisions/002-snapshot-window-design.md) | Snapshot window: `createdAt + 14d` to `endDate − 14d` |
| [ADR-003](docs/decisions/003-rolling-window-selection.md) | Rolling windows: 7d and 14d |
| [ADR-004](docs/decisions/004-outcome-threshold.md) | Outcome threshold: `yes_final_price ≥ 0.95` → YES |
| [ADR-005](docs/decisions/005-leakage-fix.md) | Remove post-resolution snapshots (price crossed 0.95/0.05) |
| [ADR-006](docs/decisions/006-svm-subsampling.md) | SVM subsampling: 7 fixed percentile snapshots per market |
| [ADR-007](docs/decisions/007-class-imbalance.md) | Class imbalance: `class_weight='balanced'` throughout |
| [ADR-008](docs/decisions/008-market-filters.md) | Market filters: volume ≥ $1k, duration ≥ 30 days |
| [ADR-009](docs/decisions/009-google-trends-enrichment.md) | Google Trends joined at category + week level |
| [ADR-010](docs/decisions/010-database-design.md) | PostgreSQL via Supabase; 5-table schema |
| [ADR-011](docs/decisions/011-offline-online-split.md) | Parquet for training; DB for live/operational layer only |
| [ADR-012](docs/decisions/012-s3-duckdb-data-lake.md) | S3 + DuckDB as the data lake; parquet removed from git |

---

## Environment

Python 3.13+.

```bash
pip install -e ".[dev]"
```

`ANTHROPIC_API_KEY` is only required for `categorize_markets.py`.
