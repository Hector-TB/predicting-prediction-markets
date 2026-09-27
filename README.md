# Predicting Prediction Markets

ML models trained on historical [Polymarket](https://polymarket.com) binary prediction market data to forecast YES/NO outcomes. The core question: **can ML beat the market's own implied probability?**

Originally a course project (NYU DS-GA 1003, team of 3); the paper is in [`docs/paper/`](docs/paper/). Now being productionized into a full-stack application.

**Team:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri

---

## Research Questions

| | Question |
|---|---|
| **RQ1** | Can ML outperform market probabilities? |
| **RQ2** | Do Google Trends features add predictive value? |
| **RQ3** | How do model predictions vary across the market lifecycle? |

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
- `ANTHROPIC_API_KEY` — only needed to re-run `categorize_markets.py` (loaded from `.env`)

See `.env.example` for the full list.

---

## Repository Structure

```
predicting-prediction-markets/
├── data/                              # parquet/CSV data files (gitignored — live on S3)
│   └── sync.py                        # push/pull data files to/from S3
├── data_collection_pipeline/          # ETL scripts — run_pipeline.py runs them in order
│   ├── run_pipeline.py                # end-to-end runner (steps 1–8)
│   ├── fetch_markets.py               # Step 1: resolved markets from Gamma API (keyset pagination)
│   ├── build_snapshots.py             # Step 2: 12-hour snapshot dataset + rolling features
│   ├── fix_dataset.py                 # Step 3: dedupe, clip prices, fill NaN features, merge to parquet
│   ├── categorize_markets.py          # Step 4: LLM category assignment (Claude API)
│   ├── fetch_category_trends.py       # Step 5: pull Google Trends data
│   ├── build_trend_features.py        # Step 6: compute trend features
│   ├── merge_trends.py                # Step 7: merge trends into snapshot dataset
│   └── fix_leakage.py                 # Step 8: remove post-close and settled snapshots
├── models/
│   ├── common/                        # shared evaluation utilities (evaluation.py)
│   ├── logistic_regression/           # train.py + notebook + predictions/
│   ├── gradient_boosting/             # XGBoost: train.py + notebook + predictions/
│   ├── random_forest/                 # train.py + notebook + predictions/
│   ├── random_forest_trends/          # train.py + notebook + predictions/
│   ├── svm/                           # svm.py + svm_evaluate.py + notebook
│   └── lightgbm/                      # placeholder (not yet implemented)
├── scripts/
│   ├── print_metrics.py               # metrics table for all models vs the market baseline
│   └── recompute_split.py             # recompute the 80/20 split after a full re-fetch
├── analysis/
│   ├── analysis.ipynb                 # model comparison, lift curves, calibration
│   ├── trading_simulation.ipynb       # simulated trading strategy from predictions
│   └── explore_data.py                # data validation + diagnostic plots → plots/
├── db/
│   ├── load_parquet.py                # migrate metadata + model registry to Supabase
│   └── migrations/                    # SQL migration files (apply via Supabase MCP)
├── docs/
│   ├── decisions/                     # Architecture Decision Records (ADR-001–016)
│   └── paper/                         # course paper (LaTeX) + notes on its results
├── plots/                             # generated PNGs (gitignored)
├── .env.example
├── pyproject.toml
└── CLAUDE.md                          # persistent context for Claude Code sessions
```

---

## Data Pipeline

All scripts resolve paths relative to `__file__`. The runner executes every step in order:

```bash
python data_collection_pipeline/run_pipeline.py                   # full incremental refresh
python data_collection_pipeline/run_pipeline.py --skip-snapshots   # skip steps 1–2
python data_collection_pipeline/run_pipeline.py --trends-only      # steps 5–8 only
```

Notes:
- `fetch_markets.py --full` re-fetches every market since 2023 with resumable checkpoints; run `scripts/recompute_split.py` afterwards (ADR-013).
- `build_snapshots.py` resumes from where it stopped; a full rebuild takes many hours.
- `categorize_markets.py` needs `ANTHROPIC_API_KEY` and only sends markets that aren't categorized yet.
- `fix_leakage.py` drops snapshots after each market's API `closedTime` and any snapshot priced ≥ 0.95 or ≤ 0.05 (ADR-005, ADR-014).

Datasets are versioned on S3 (ADR-016). Each version is immutable and has a manifest recording its date coverage, the code that built it, row counts and file checksums:
```bash
python data/sync.py list                      # versions on S3 (v1 = the course paper's data)
python data/sync.py status                    # which version is checked out locally
python data/sync.py pull --version v1         # download a version (default: LATEST)
python data/sync.py publish v2 --parent v1 --notes "…"   # after a pipeline run: new version
```

---

## Training Models

```bash
python models/logistic_regression/train.py      # add --trends for the Google Trends variant
python models/gradient_boosting/train.py        # add --trends for the Google Trends variant
python models/random_forest/train.py
python models/random_forest_trends/train.py
python models/svm/svm.py && python models/svm/svm_evaluate.py

python scripts/print_metrics.py      # all models vs the market baseline
python analysis/rescore_paper.py     # the paper's comparisons: shared test rows, market-level CIs (ADR-015)
```

All models train on the leakage-filtered `_clean` parquet files. Shared evaluation utilities (AUC-ROC, PR-AUC, log-loss, Brier, calibration) are in `models/common/evaluation.py`.

---

## Results (course paper, test set of 288,490 snapshots across 4,174 markets)

> These are the results reported in the course paper, on the v1 dataset. The models are being retrained on the expanded v2 dataset, and this section will be updated with the new results.

The baseline is the market price itself (`price_at_snapshot` as a probability), scored on the same test set.

| Model | AUC-ROC | PR-AUC | Log-loss | Brier |
|---|---|---|---|---|
| Market price (baseline) | 0.8890 | 0.7440 | 0.3278 | 0.1004 |
| Logistic regression | 0.8874 | 0.7101 | 0.3317 | 0.0980 |
| Logistic regression + Trends | 0.8868 | 0.7096 | 0.3301 | 0.0980 |
| XGBoost | 0.9000 | 0.7445 | 0.3073 | 0.0941 |
| XGBoost + Trends | 0.9019 | 0.7525 | 0.3043 | 0.0930 |
| Random forest | 0.9018 | 0.7527 | 0.3027 | 0.0922 |
| **Random forest + Trends** | **0.9034** | **0.7598** | **0.3011** | **0.0914** |

SVM (AUC ≈ 0.94) was scored on a different subsample (7 lifetime-percentile snapshots per market) and isn't comparable to the rows above.

**Key findings (paper):**
- **RQ1:** Tree models beat the market on every metric (best: +0.014 AUC, −0.027 log-loss). Logistic regression doesn't, which suggests the signal comes from non-linear interactions between price, lifecycle position and volatility.
- **RQ2:** Google Trends adds small, consistent gains for tree models (+0.001–0.002 AUC), mostly in geopolitics and finance.
- **RQ3:** The models add the most early in a market's life (+0.016–0.019 AUC over the market in the first third, shrinking to +0.003–0.006 in the last third).
- **Trading simulation:** trading on the gap between model and market price was profitable in backtests, ahead of an always-buy-NO strategy (no fees or slippage modelled). Updated figures will follow the v2 re-score.

---

## Dataset

> The figures in this section describe the v1 dataset (20,948 markets). The ADR-013 full re-fetch expanded the market list to 45,143 markets, and the snapshot dataset is being rebuilt from it; the counts below will be updated once the rebuild finishes.

### Canonical files (on S3 — fetch with `python data/sync.py pull`; see ADR-016 for versions)

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

`001_initial_schema.sql` already includes the `model_runs` metrics and hyperparameter JSONB columns that `004_add_model_run_metrics.sql` adds.

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
| [ADR-005](docs/decisions/005-leakage-fix.md) | Remove post-resolution snapshots (revised by ADR-014) |
| [ADR-006](docs/decisions/006-svm-subsampling.md) | SVM subsampling: 7 fixed percentile snapshots per market |
| [ADR-007](docs/decisions/007-class-imbalance.md) | Class imbalance: `class_weight='balanced'` throughout |
| [ADR-008](docs/decisions/008-market-filters.md) | Market filters: volume ≥ $1k, duration ≥ 30 days |
| [ADR-009](docs/decisions/009-google-trends-enrichment.md) | Google Trends joined at category + week level |
| [ADR-010](docs/decisions/010-database-design.md) | PostgreSQL via Supabase; 5-table schema |
| [ADR-011](docs/decisions/011-offline-online-split.md) | Parquet for training; DB for live/operational layer only |
| [ADR-012](docs/decisions/012-s3-duckdb-data-lake.md) | S3 + DuckDB as the data lake; parquet removed from git |
| [ADR-013](docs/decisions/013-keyset-pagination-full-refetch.md) | Keyset pagination for market fetch; full re-fetch with recomputed split |
| [ADR-014](docs/decisions/014-leakage-filter-closedtime-and-settled-rows.md) | Leakage filter: working `closedTime` lookup; drop snapshots priced ≥ 0.95 / ≤ 0.05 |
| [ADR-015](docs/decisions/015-evaluation-methodology.md) | Evaluation: shared test rows, market-level bootstrap CIs, NO trades cost 1 − p |
| [ADR-016](docs/decisions/016-dataset-versioning.md) | Immutable, manifest-described dataset versions on S3 (`v1`, `v2`, …) |

---

## Environment

Python 3.13+.

```bash
pip install -e ".[dev]"
```

`ANTHROPIC_API_KEY` is only required for `categorize_markets.py`.
