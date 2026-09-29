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
│   ├── manifests/                     # one JSON manifest per published dataset version
│   └── sync.py                        # list / status / pull / publish dataset versions (S3)
├── data_collection_pipeline/          # ETL scripts — run_pipeline.py runs them in order
│   ├── run_pipeline.py                # end-to-end runner (metadata check + steps 1–8)
│   ├── meta_checks.py                 # market metadata must match Gamma (ADR-023); gates build + publish
│   ├── fetch_markets.py               # Step 1: resolved markets from Gamma API (keyset pagination)
│   ├── build_snapshots.py             # Step 2: 12-hour snapshot dataset + rolling features (4 worker processes)
│   ├── fix_dataset.py                 # Step 3: dedupe, clip prices, fill NaN features, merge to parquet
│   ├── categorize_markets.py          # Step 4: LLM category assignment (Claude API)
│   ├── fetch_category_trends.py       # Step 5: pull Google Trends data
│   ├── build_trend_features.py        # Step 6: compute trend features
│   ├── merge_trends.py                # Step 7: merge trends into snapshot dataset
│   ├── fix_leakage.py                 # Step 8: remove post-close and settled snapshots; mark pre-cutoff test rows
│   └── stream_parquet.py              # batch-at-a-time parquet I/O (the dataset doesn't fit in 8 GB RAM)
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
│   ├── recompute_split.py             # 80/20 split by resolution time (ADR-021)
│   ├── refresh_meta.py                # repair market metadata from Gamma (ADR-023)
│   ├── coverage_check.py              # are any qualifying markets missing? (by end date, ADR-020)
│   ├── check_snapshots.py             # full quality check of a build before publishing
│   ├── stage_backup.py                # back up an unpublished build to S3 staging
│   └── smoke_test.py                  # quick end-to-end sanity check
├── analysis/
│   ├── analysis.ipynb                 # model comparison, lift curves, calibration
│   ├── trading_simulation.ipynb       # simulated trading strategy from predictions
│   └── explore_data.py                # data validation + diagnostic plots → plots/
├── db/
│   ├── load_parquet.py                # migrate metadata + model registry to Supabase
│   └── migrations/                    # SQL migration files (apply via Supabase MCP)
├── docs/
│   ├── decisions/                     # Architecture Decision Records (ADR-001–023)
│   ├── design/                        # design docs (search-and-predict site)
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
python data_collection_pipeline/run_pipeline.py --skip-markets --rebuild-snapshots   # rebuild every market's snapshots
python data_collection_pipeline/run_pipeline.py --skip-trends      # reuse the saved Google Trends data
```

Notes:
- **Market metadata must match Gamma exactly** (ADR-023). Only `category` and `split` are ours; nothing is estimated or back-filled. `meta_checks.py` compares every field with the saved Gamma records, and the pipeline, the snapshot build and `publish` refuse to run if it fails. Repair with `scripts/refresh_meta.py`.
- `fetch_markets.py --full` re-fetches every market since 2023 with resumable checkpoints (ADR-013). Fresh Gamma values always replace stored ones. Run `scripts/recompute_split.py` afterwards.
- `build_snapshots.py` builds snapshots from 14 days after creation up to the scheduled end, stopping at the market's close (ADR-002, ADR-022). It runs 4 worker processes (a full rebuild takes about 5–6 hours), retries failed requests, and resumes where it stopped.
- `categorize_markets.py` needs `ANTHROPIC_API_KEY` and only sends markets without a valid category.
- `fetch_category_trends.py` re-fetches the whole Trends range each run and keeps the saved file if Google rate-limits it (ADR-009).
- `fix_leakage.py` drops snapshots after each market's API `closedTime` and any snapshot priced ≥ 0.95 or ≤ 0.05 (ADR-014). It marks test rows dated before the train/test cutoff as `test_pre_cutoff` (ADR-021).

Before publishing a new version, run both checks:
```bash
python scripts/coverage_check.py      # qualifying markets missing from the fetch? (exits non-zero if any)
python scripts/check_snapshots.py     # full quality check of the build, compared with the latest version
```

Datasets are versioned on S3 (ADR-016). Each version is immutable and has a manifest (`data/manifests/`) recording its coverage, the code that built it, row counts and file checksums:

| Version | What it is |
|---|---|
| `v1` | The course project's dataset, used by the paper |
| `v2` | Full re-fetch (ADR-013), working leakage filter (ADR-014), split by resolution time (ADR-021) |
| `v3` (latest) | v2 without the 14-day pre-close cutoff (ADR-022), metadata matched to Gamma (ADR-023) |

```bash
python data/sync.py list                      # versions on S3
python data/sync.py status                    # which version is checked out locally
python data/sync.py pull --version v1         # download a version (default: LATEST)
python data/sync.py publish v4 --parent v3 --notes "…"   # after a pipeline run + checks: new version
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

> These are the results reported in the course paper, on the v1 dataset. The models are being retrained on the current dataset (v3), and this section will be updated with the new results.

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
- **Trading simulation:** trading on the gap between model and market price was profitable in backtests, ahead of an always-buy-NO strategy (no fees or slippage modelled). Updated figures will follow the v3 re-score.

---

## Dataset

> The figures in this section describe the current dataset, **v3**. The results above are the paper's, on v1 (20,948 markets, 1,448,142 snapshots).

### Canonical files (on S3 — fetch with `python data/sync.py pull`; see ADR-016 for versions)

| File | Rows | Description |
|---|---|---|
| `polymarket_ml_dataset_clean.parquet` | 2,534,852 | Base features — use for training |
| `polymarket_ml_dataset_with_trends_clean.parquet` | 2,534,852 | Base + Google Trends features |
| `category_trends_features.parquet` | 1,755 | Weekly trend aggregates by category |
| `polymarket_markets_meta.csv` | 45,130 | Market metadata (regenerate with `fetch_markets.py`) |

### Markets (`polymarket_markets_meta.csv`)

Dataset v3: 45,130 resolved binary Polymarket markets whose Polymarket start date is 2023 or later. Filters: volume ≥ $1,000, duration ≥ 30 days, clear resolution (ADR-004, ADR-008). Every field except `category` and `split` is Gamma's own value (ADR-023).

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
| `split` | `"train"` or `"test"` (market-level 80/20 by resolution time, ADR-021) |

### Snapshots (`polymarket_ml_dataset_clean.parquet`)

One row per (market × 12-hour snapshot), from 14 days after creation to the market's close, minus near-certain rows (price ≥ 0.95 or ≤ 0.05). Dataset v3: 2,534,852 rows across 29,531 markets, 27 columns.

Feature groups:
- **Time:** `days_before_close`, `pct_lifetime_elapsed`, `duration_days`
- **Price:** `price_at_snapshot`, `price_deviation_from_half`
- **Rolling 7d:** `price_mean_7d`, `price_volatility_7d`, `price_min_7d`, `price_max_7d`, `price_change_7d`, `price_range_7d`, `price_trend_7d`
- **Rolling 14d:** same set with `_14d` suffix
- **Volume:** `total_volume`, `log_volume` — in the data but not used by any model: they hold the market's *final* lifetime volume, which isn't known at snapshot time (ADR-018)
- **Target:** `outcome`

### Train / Test Split

Markets are split by **when they resolved** (ADR-021). The first 80% to resolve are train; the rest are test. The cutoff T is the last train resolution (v3: 2026-07-01 07:11 UTC). Every training label was known by T, and only test snapshots dated T or later are scored. Earlier test rows are marked `test_pre_cutoff` and unused. No market appears in both splits.

| Split (v3) | Markets | Snapshots | YES rate (rows) |
|---|---|---|---|
| Train | 23,120 | 2,038,895 | 25.4% |
| Test | 4,924 | 196,977 | 38.4% |
| `test_pre_cutoff` (unused) | 3,233 | 298,980 | 30.5% |

The test set only contains markets that have resolved, so it leans towards shorter markets (a selection effect, not a leak). Of the 45,130 markets, 29,531 have usable snapshots: the rest have no price history, too little history, or only near-certain prices.

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

Each snapshot uses the most recent *completed* week (ADR-009 amendment). Coverage: 99.3% of v3 snapshots. See ADR-009.

### Categories (LLM-assigned via Claude API)

| Category | Markets | % |
|---|---|---|
| `sports` | 14,728 | 32.6% |
| `politics_us` | 5,997 | 13.3% |
| `finance` | 5,989 | 13.3% |
| `entertainment` | 5,162 | 11.4% |
| `crypto` | 3,664 | 8.1% |
| `politics_global` | 3,636 | 8.1% |
| `science_tech` | 2,898 | 6.4% |
| `geopolitics` | 2,531 | 5.6% |
| `other` | 525 | 1.2% |

---

## Database

Supabase Postgres (operational layer — not used for training). Schema: `markets`, `snapshots`, `trends`, `model_runs`, `predictions`. Parquet files are the training data store; the DB serves live market scoring and the future frontend.

```bash
# Populate the DB (runs in seconds — metadata + model registry only; re-running updates existing rows)
python db/load_parquet.py
```

`001_initial_schema.sql` already includes the `model_runs` metrics and hyperparameter JSONB columns that `004_add_model_run_metrics.sql` adds.

See ADR-010 (database design) and ADR-011 (offline/online split).

---

## Architecture Decision Records

All significant decisions are documented in `docs/decisions/`. These are the audit trail for every methodological choice.

| ADR | Decision |
|---|---|
| [ADR-001](docs/decisions/001-temporal-train-test-split.md) | Temporal, market-level 80/20 train/test split (split rule superseded by ADR-021) |
| [ADR-002](docs/decisions/002-snapshot-window-design.md) | Snapshot window: `createdAt + 14d` to `endDate − 14d` (end cutoff removed from v3, ADR-022) |
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
| [ADR-017](docs/decisions/017-on-demand-prediction-live-track-record.md) | On-demand prediction via search, with a stored live track record |
| [ADR-018](docs/decisions/018-volume-feature-look-ahead.md) | Volume features: final lifetime volume is look-ahead — dropped from the models |
| [ADR-019](docs/decisions/019-model-releases-promotion-rollback.md) | Immutable model releases, change reports, gated promotion, rollback |
| [ADR-020](docs/decisions/020-incremental-fetch-by-end-date.md) | Incremental market fetch by scheduled end date, with a look-back; coverage check |
| [ADR-021](docs/decisions/021-leak-free-evaluation.md) | Leak-free evaluation: split by resolution time, no tuning on the test set |
| [ADR-022](docs/decisions/022-remove-pre-close-cutoff.md) | No 14-day pre-close cutoff (dataset v3); full rebuild with retries |
| [ADR-023](docs/decisions/023-meta-must-match-gamma.md) | Market metadata must match Gamma; checks block build and publish |
| [ADR-024](docs/decisions/024-shared-training-protocol.md) | One training protocol for all models: time-ordered holdout, one weight per market, calibration |

---

## Environment

Python 3.13+.

```bash
pip install -e ".[dev]"
```

`ANTHROPIC_API_KEY` is only required for `categorize_markets.py`.
