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

| Plot | What it shows |
|---|---|
| `class_imbalance.png` | YES/NO bar chart at both market level (19.7% YES) and snapshot level (22.1% YES) — visualises the class imbalance models must handle |
| `meta_duration_hist.png` | Histogram of market duration in days (median 89d); shows the long tail of multi-year markets |
| `meta_volume_hist.png` | Log-scale volume distribution; 10.3% of markets trade > $1M, 24.9% trade < $10k |
| `dataset_price_distribution.png` | Overlaid density of `price_at_snapshot` for YES vs NO outcomes — shows how well-separated the two distributions are even mid-market |
| `dataset_snapshots_per_market.png` | Distribution of snapshots per market (median 29, max 885); highlights the imbalance in how much each market contributes |
| `feature_nan_by_lifecycle.png` | NaN rate in 7d rolling features by lifecycle decile — NaNs appear only in the earliest 10% of a market's life and are filled before modelling |
| `calibration_curve.png` | Binned calibration plot of `price_at_snapshot` vs realised YES fraction on the train split — confirms near-perfect market calibration (AUC-ROC = 0.964) |

**Baseline (train split):** AUC-ROC = 0.964, Log-loss = 0.171, Accuracy = 0.938  
The market price alone is a near-perfect predictor. Any ML model must beat AUC-ROC > 0.964 or log-loss < 0.171 to demonstrate added value (RQ1).

---

## Dataset

### `data/polymarket_markets_meta.csv`
One row per market. **24,092 markets** after deduplication.

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

**Duration distribution:**

| Range | Count | % |
|---|---|---|
| 30–60 days | 8,749 | 36.3% |
| 60–90 days | 3,637 | 15.1% |
| 90–180 days | 5,895 | 24.5% |
| 180–365 days | 5,104 | 21.2% |
| 365+ days | 707 | 2.9% |

**Volume distribution:**

| Range | Count | % |
|---|---|---|
| $1k – $10k | 6,010 | 24.9% |
| $10k – $100k | 9,513 | 39.5% |
| $100k – $1M | 6,088 | 25.3% |
| $1M+ | 2,481 | 10.3% |

### `data/polymarket_ml_dataset_clean.parquet` ← use this for modelling
One row per (market × 12-hour snapshot). **1,448,142 rows, 27 columns.**  
This is the leakage-fixed version produced by `fix_leakage.py`. The raw pre-fix file (`polymarket_ml_dataset.parquet`) has ~4.1M rows; the ~2.65M removed rows are post-resolution snapshots where markets had already effectively settled (price crossed 0.95/0.05) and were just accumulating stale observations.

Feature groups:
- **Time:** `days_before_close`, `pct_lifetime_elapsed`, `duration_days`
- **Price:** `price_at_snapshot`, `price_deviation_from_half`
- **Rolling 7d & 14d:** `price_mean`, `price_volatility`, `price_min`, `price_max`, `price_change`, `price_range`, `price_trend`
- **Volume:** `total_volume`, `log_volume`
- **Target:** `outcome`

### Feature Engineering Rationale

**Why 7-day and 14-day windows?**  
Prediction markets typically exhibit momentum and mean-reversion effects on the scale of days to weeks. A 7-day window captures short-term momentum (e.g., a news event shifting the price this week); 14 days captures slower trend drift and provides a more stable baseline to compare against. Using two windows lets models detect *divergence* between short and medium-term signals — for example, a price that is rising over 7 days but still below its 14-day mean.

**Why those rolling statistics?**
- `price_mean` — smoothed probability estimate; less noisy than the current price alone
- `price_volatility` (std) — market uncertainty; high volatility near a deadline is a signal the market is unsettled
- `price_min` / `price_max` — range endpoints; useful for detecting whether a price move is a new extreme or a recovery
- `price_change` — net drift over the window; captures directional momentum
- `price_range` — difference of max and min; width of oscillation independent of direction
- `price_trend` — linear slope of prices over the window; distinguishes a steady drift from a noisy flat market

**Feature correlations with outcome (train split):**

| Feature | Correlation with outcome |
|---|---|
| `price_at_snapshot` | +0.643 |
| `price_mean_7d` | +0.637 |
| `price_mean_14d` | +0.627 |
| `price_deviation_from_half` | −0.226 |
| `price_change_14d` | +0.217 |
| `price_trend_14d` | +0.157 |
| `price_change_7d` | +0.136 |
| `price_trend_7d` | +0.120 |
| `price_volatility_14d` | +0.109 |
| `log_volume` | +0.102 |
| `price_volatility_7d` | +0.099 |
| `pct_lifetime_elapsed` | +0.073 |
| `duration_days` | +0.023 |
| `days_before_close` | −0.011 |

The current price dominates; rolling means are nearly as informative (they smooth the same signal). Momentum (`price_change`, `price_trend`) and volume carry modest independent signal. `days_before_close` is nearly uncorrelated with outcome, confirming that predicting late is not inherently easier.

### Class Imbalance

At the market level, **80.3% of markets resolve NO** and only 19.7% resolve YES. In the snapshot dataset the imbalance is slightly less severe (77.9% NO / 22.1% YES) because YES markets tend to be longer-lived and therefore produce more snapshots.

This is a significant modelling challenge. A trivial classifier that always predicts NO achieves ~78% accuracy — making raw accuracy a misleading metric. **Use AUC-ROC and log-loss as primary metrics.** Models should use `class_weight='balanced'` or equivalent to avoid learning to always predict the majority class.

### Train / Test Split

The split is assigned at the **market level** before any snapshots are generated, so no snapshot from a test market ever appears in training. The cutoff is temporal — the oldest 80% of markets by creation date form the train set.

| Split | Markets | Snapshots | YES rate |
|---|---|---|---|
| Train | 16,774 | 1,159,652 | 22.4% |
| Test | 4,174 | 288,490 | 21.0% |
| **Total** | **20,948** | **1,448,142** | **22.1%** |

### Dropped Markets

Of 24,092 markets in the meta CSV, **3,144 (13.0%) are absent from the snapshot dataset**. Reasons (applied in order during `build_snapshots.py`):
1. Fewer than 20 price observations in the CLOB API — too sparse to compute rolling features
2. Valid snapshot window (`createdAt + 14d` to `endDate − 14d`) collapses to zero — typically markets that resolved early or had a very short effective trading period

These dropped markets are not systematically biased by category or volume, so exclusion is unlikely to introduce selection bias.

### Google Trends Enrichment

`data/polymarket_ml_dataset_with_trends_clean.parquet` adds 5 columns per snapshot:

| Column | Description |
|---|---|
| `trend_value` | Weekly Google Trends interest score (0–100) for the market's category |
| `trend_ma4` | 4-week moving average of `trend_value` |
| `trend_change_4w` | `trend_value` minus value 4 weeks prior |
| `trend_spike` | 1 if `trend_value > 1.5 × trend_ma4`, else 0 |
| `has_trend_data` | 1 if trend data exists for this category, 0 otherwise |

**Trends coverage: 99.6%** of snapshots have `has_trend_data = 1`. The 0.4% without trend data (6,424 snapshots) are almost entirely from the `other` category (6,347 snapshots), which has no defined keyword set, plus a small number of early `politics_us` snapshots that predate the trends data range. These rows get `trend_value = 0`, which is a safe neutral fill.

---

## Categories (LLM-assigned)

Market questions are classified into 9 categories using the Claude API (`claude-haiku-4-5`, batches of 100). Classification is idempotent — only uncategorised or `"other"` markets are re-processed on rerun.

| Category | Count | % | Description |
|---|---|---|---|
| `sports` | 7,081 | 29.4% | NFL, NBA, MLB, NHL, soccer, tennis, any sport |
| `politics_us` | 3,751 | 15.6% | US elections, congress, presidency, domestic policy |
| `entertainment` | 3,553 | 14.7% | Celebrity, TV, music, film, pop culture |
| `finance` | 2,846 | 11.8% | Fed, inflation, stocks, IPOs, economic indicators |
| `crypto` | 2,367 | 9.8% | Bitcoin, ETH, DeFi, tokens, NFTs, blockchain |
| `politics_global` | 1,900 | 7.9% | Foreign elections, heads of state, international politics |
| `geopolitics` | 1,292 | 5.4% | Wars, sanctions, treaties, military conflicts |
| `science_tech` | 1,149 | 4.8% | AI, space, FDA approvals, climate, tech products |
| `other` | 152 | 0.6% | Anything else |

Sports dominates at nearly 30% of all markets, reflecting Polymarket's heavy user base for sports betting. Category is used both as a stratification variable in analysis and as the join key for Google Trends enrichment.

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
