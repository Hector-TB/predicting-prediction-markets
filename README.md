# Predicting the Future: ML on Prediction Markets

**DS-GA 1003 Machine Learning — NYU, Applied Track**  
**Team:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri

---

## Overview

We train ML models on [Polymarket](https://polymarket.com) prediction market data to forecast binary (YES/NO) outcomes, and investigate whether models can beat the market's own implied probability. We also examine whether engineered price features (momentum, volatility, trend) add predictive value beyond the market price alone, and at which stage of a market's lifecycle external signals are most useful.

### Research Questions

| | Question |
|---|---|
| **RQ1** | Can ML models outperform prediction market probabilities in forecasting outcomes? |
| **RQ2** | Do engineered features beyond market price enhance predictive performance? |
| **RQ3** | At which stage of a market's lifecycle do external signals provide the most value? |

---

## Repository Structure

```
predicting-the-future/
├── pipeline/               # Data collection and processing
│   ├── fetch_markets.py    # Step 1: fetch markets from Gamma API
│   ├── update_markets.py   # Step 1b: incremental market update
│   ├── build_snapshots.py  # Step 2: fetch CLOB price history + build snapshot dataset
│   ├── fix_dataset.py      # Step 3: deduplicate, clip prices, fill NaN features
│   └── categorize_markets.py  # Step 4: LLM-based category assignment (Claude API)
├── analysis/
│   └── explore_data.py     # Data validation and EDA — produces plots/
├── notebooks/
│   └── testing.ipynb       # Scratch notebook
├── data/                   # gitignored — CSV and parquet data files
├── plots/                  # gitignored — output figures from explore_data.py
└── README.md
```

---

## Data Pipeline

Run steps in order. Each script resolves paths relative to itself, so it works from any working directory.

### Step 1 — Fetch Markets
```bash
python3 pipeline/fetch_markets.py
```
Hits the Polymarket Gamma API, filters for resolved binary markets with volume >= $1,000 and duration >= 30 days, assigns an 80/20 train/test split at market level, and writes `data/polymarket_markets_meta.csv`.

### Step 1b — Incremental Update (optional)
```bash
python3 pipeline/update_markets.py
```
Reads the most recent `start_date` from the meta CSV and appends only newer markets. Safe to rerun.

### Step 2 — Build Snapshot Dataset
```bash
python3 pipeline/build_snapshots.py
```
For each market, fetches 12-hour price history from the CLOB API and generates one row per snapshot with rolling price features (7d and 14d windows). Snapshot window: `createdAt + 14 days` to `endDate - 14 days`. Supports checkpoint/resume. Writes `data/polymarket_ml_dataset.csv`.

### Step 3 — Fix Dataset
```bash
python3 pipeline/fix_dataset.py
```
Applies three data quality fixes in place:
1. Deduplicate market IDs in meta CSV
2. Clip `price_at_snapshot` to `[0, 1]`
3. Fill NaN rolling features (early snapshots with no trading history) — uses `price_at_snapshot` for mean/min/max and `0` for volatility/change/range/trend

### Step 4 — Categorise Markets (optional, requires Claude API key)
```bash
ANTHROPIC_API_KEY=... python3 pipeline/categorize_markets.py
```
Uses the Claude API to classify each market question into one of 9 categories. Safe to rerun — only re-classifies missing or "other" rows.

### Convert to Parquet (recommended)
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/polymarket_ml_dataset.csv', low_memory=False)
df.to_parquet('data/polymarket_ml_dataset.parquet', index=False)
"
```
Reduces the dataset from ~1.2 GB CSV to ~88 MB parquet (13x compression).

---

## Exploration & Validation

```bash
python3 analysis/explore_data.py
```

Reads `data/polymarket_markets_meta.csv` and `data/polymarket_ml_dataset.parquet`. Produces text output with data quality checks and saves 7 diagnostic plots to `plots/`:

| Plot | Description |
|---|---|
| `class_imbalance.png` | YES/NO counts at market and snapshot level |
| `meta_duration_hist.png` | Market duration distribution |
| `meta_volume_hist.png` | Total volume distribution (log scale) |
| `dataset_price_distribution.png` | Price at snapshot by final outcome |
| `dataset_snapshots_per_market.png` | Snapshots per market distribution |
| `feature_nan_by_lifecycle.png` | NaN rate in rolling features by lifecycle stage |
| `calibration_curve.png` | Market price calibration curve with AUC-ROC and log-loss baseline |

**Baseline (train split):** AUC-ROC = 0.964, Log-loss = 0.171, Accuracy = 0.938

---

## Dataset

### `data/polymarket_markets_meta.csv`
One row per market (~24,092 markets after deduplication).

| Column | Description |
|---|---|
| `market_id` | Polymarket condition ID |
| `clob_token_id` | YES token ID for CLOB API |
| `question` | Raw market question text |
| `category` | LLM-assigned category (9 classes) |
| `start_date` | Market creation date |
| `end_date` | Scheduled close date |
| `duration_days` | `end_date - start_date` in days |
| `total_volume` | Lifetime trading volume (USD) |
| `yes_final_price` | Final settlement price of YES token |
| `outcome` | Ground truth: 1 = YES, 0 = NO |
| `split` | `"train"` or `"test"` (market-level 80/20) |

### `data/polymarket_ml_dataset.parquet`
One row per (market x 12-hour snapshot). ~4.1M rows, 27 columns.

Feature groups:
- **Time:** `days_before_close`, `pct_lifetime_elapsed`, `duration_days`
- **Price:** `price_at_snapshot`, `price_deviation_from_half`
- **Rolling 7d & 14d:** `price_mean`, `price_volatility`, `price_min`, `price_max`, `price_change`, `price_range`, `price_trend`
- **Volume:** `total_volume`, `log_volume`
- **Target:** `outcome`

### Key Statistics

| Metric | Value |
|---|---|
| Markets (meta) | 24,092 |
| Markets (dataset) | 21,316 |
| Total snapshots | 4,078,109 |
| Outcome balance | 80.3% NO / 19.7% YES |
| Date range | 2023-01-01 to 2026-02-06 |
| Train / test split | 80% / 20% (market-level, zero overlap) |

---

## Categories (LLM-assigned)

| Category | Description |
|---|---|
| `politics_us` | US elections, congress, presidency, domestic policy |
| `politics_global` | Foreign elections, heads of state, international politics |
| `crypto` | Bitcoin, ETH, DeFi, tokens, NFTs, blockchain |
| `sports` | NFL, NBA, MLB, NHL, soccer, tennis, any sport |
| `finance` | Fed, inflation, stocks, IPOs, economic indicators |
| `geopolitics` | Wars, sanctions, treaties, military conflicts |
| `science_tech` | AI, space, FDA approvals, climate, tech products |
| `entertainment` | Celebrity, TV, music, film, pop culture |
| `other` | Anything else |

---

## Models (planned)

1. Logistic Regression (L1/L2 regularisation)
2. SVM (linear and RBF kernels)
3. Decision Tree
4. Random Forest
5. Gradient Boosting

Evaluation: AUC-ROC, log-loss, accuracy, calibration plots. 5-fold cross-validation stratified by outcome, grouped by market.

---

## Environment

```bash
pip install pandas numpy requests scikit-learn anthropic pyarrow matplotlib seaborn
```

Python 3.13+. `ANTHROPIC_API_KEY` is only required for `categorize_markets.py`.
