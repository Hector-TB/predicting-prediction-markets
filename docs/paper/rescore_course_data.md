<!-- Generated 2026-09-27 by analysis/rescore_paper.py on the course-era data (old polymarket_ml_dataset_clean.parquet + course prediction files), to validate the script against the paper. -->

# Paper re-score — polymarket_ml_dataset_clean.parquet

## RQ1 — models vs the market price

288,490 test snapshots, 4,174 markets. ΔAUC = model − market, 95% CI from 200 market-level bootstrap resamples.

| Model | AUC-ROC | PR-AUC | Log-loss | Brier | ΔAUC vs market [95% CI] |
|---|---|---|---|---|---|
| Market price | 0.8890 | 0.7440 | 0.3278 | 0.1004 | — |
| LR | 0.8874 | 0.7101 | 0.3301 | 0.0980 | -0.0016 [-0.0076, +0.0038] |
| LR + Trends | 0.8868 | 0.7096 | 0.3289 | 0.0980 | -0.0022 [-0.0086, +0.0031] |
| GB | 0.9000 | 0.7445 | 0.3069 | 0.0941 | +0.0110 [+0.0029, +0.0212] |
| GB + Trends | 0.9019 | 0.7525 | 0.3040 | 0.0930 | +0.0129 [+0.0047, +0.0220] |
| RF | 0.9018 | 0.7527 | 0.3026 | 0.0922 | +0.0128 [+0.0057, +0.0196] |
| RF + Trends | 0.9034 | 0.7598 | 0.3008 | 0.0914 | +0.0143 [+0.0054, +0.0231] |

## RQ3 — lifecycle stage

AUC-ROC per bucket.

| Bucket | Snapshots | Market price | LR | LR + Trends | GB | GB + Trends | RF | RF + Trends |
|---|---|---|---|---|---|---|---|---|
| Far (0–33%) | 137,054 | 0.8910 | 0.8867 | 0.8865 | 0.9048 | 0.9069 | 0.9054 | 0.9064 |
| Mid (33–67%) | 119,722 | 0.8830 | 0.8838 | 0.8825 | 0.8936 | 0.8950 | 0.8963 | 0.8997 |
| Near (67–100%) | 31,714 | 0.8981 | 0.8962 | 0.8959 | 0.8998 | 0.9024 | 0.9028 | 0.8990 |

## RQ3 — market duration

AUC-ROC per bucket.

| Bucket | Snapshots | Market price | LR | LR + Trends | GB | GB + Trends | RF | RF + Trends |
|---|---|---|---|---|---|---|---|---|
| ≤ 90d | 65,747 | 0.9146 | 0.9194 | 0.9191 | 0.9178 | 0.9213 | 0.9247 | 0.9262 |
| 91–180d | 94,751 | 0.8650 | 0.8686 | 0.8682 | 0.8653 | 0.8686 | 0.8734 | 0.8703 |
| 181–365d | 116,462 | 0.8821 | 0.8741 | 0.8728 | 0.9051 | 0.9044 | 0.9028 | 0.9070 |
| > 365d | 11,530 | 0.9698 | 0.9625 | 0.9635 | 0.9776 | 0.9801 | 0.9769 | 0.9791 |

## Trading simulation (paper §5.5)

Trade when |p_model − p_market| > 0.02; one token per trade; no fees or slippage. ROI = total profit / total capital; 95% CI from market-level bootstrap.

### Per-snapshot (paper's method)

| Strategy | Trades | Markets | Profit | ROI [95% CI] | Win rate |
|---|---|---|---|---|---|
| Always buy NO | 288,490 | 4,174 | $15,423 | 7.3% [4.9%, 9.2%] | 79.0% |
| LR | 219,973 | 3,572 | $10,670 | 7.4% [4.8%, 10.0%] | 70.4% |
| LR + Trends | 218,328 | 3,536 | $10,691 | 7.5% [4.9%, 9.8%] | 70.5% |
| GB | 215,490 | 3,241 | $16,757 | 11.9% [9.6%, 14.7%] | 72.9% |
| GB + Trends | 217,570 | 3,248 | $18,097 | 12.7% [10.5%, 15.5%] | 73.8% |
| RF | 199,506 | 3,230 | $15,047 | 11.4% [9.2%, 13.6%] | 73.8% |
| RF + Trends | 204,732 | 3,191 | $17,004 | 13.1% [11.4%, 14.7%] | 71.6% |

### One trade per market (first qualifying snapshot)

| Strategy | Trades | Markets | Profit | ROI [95% CI] | Win rate |
|---|---|---|---|---|---|
| Always buy NO | 4,174 | 4,174 | $185 | 5.7% [4.7%, 6.7%] | 82.9% |
| LR | 3,572 | 3,572 | $168 | 8.6% [6.7%, 10.4%] | 59.8% |
| LR + Trends | 3,536 | 3,536 | $166 | 8.5% [6.6%, 10.2%] | 60.2% |
| GB | 3,241 | 3,241 | $195 | 10.0% [8.3%, 11.7%] | 66.5% |
| GB + Trends | 3,248 | 3,248 | $227 | 11.2% [9.6%, 13.3%] | 69.2% |
| RF | 3,230 | 3,230 | $196 | 10.0% [8.2%, 11.9%] | 66.4% |
| RF + Trends | 3,191 | 3,191 | $183 | 11.1% [8.8%, 13.4%] | 57.7% |
