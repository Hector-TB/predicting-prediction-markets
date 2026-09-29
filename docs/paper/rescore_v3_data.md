# Paper re-score — polymarket_ml_dataset_clean.parquet

## RQ1 — models vs the market price

196,977 test snapshots, 4,924 markets. ΔAUC = model − market, 95% CI from 500 market-level bootstrap resamples.

| Model | AUC-ROC | PR-AUC | Log-loss | Brier | ΔAUC vs market [95% CI] |
|---|---|---|---|---|---|
| Market price | 0.7986 | 0.7390 | 0.5171 | 0.1727 | — |
| LR | 0.8014 | 0.7281 | 0.5223 | 0.1730 | +0.0029 [-0.0012, +0.0069] |
| GB | 0.8127 | 0.7414 | 0.5105 | 0.1692 | +0.0141 [+0.0092, +0.0203] |
| RF | 0.8122 | 0.7374 | 0.5097 | 0.1692 | +0.0136 [+0.0090, +0.0194] |

## RQ3 — lifecycle stage (thirds of each market's snapshots, as in the paper's figure)

AUC-ROC per bucket.

| Bucket | Snapshots | Markets | Market price | LR | GB | RF |
|---|---|---|---|---|---|---|
| Far (0–33%) | 67,444 | 4,924 | 0.7508 | 0.7597 | 0.7726 | 0.7716 |
| Mid (33–67%) | 65,646 | 4,683 | 0.7938 | 0.7955 | 0.8078 | 0.8073 |
| Near (67–100%) | 63,887 | 4,545 | 0.8473 | 0.8469 | 0.8558 | 0.8557 |

## RQ3 — lifecycle stage (by pct_lifetime_elapsed)

AUC-ROC per bucket.

| Bucket | Snapshots | Markets | Market price | LR | GB | RF |
|---|---|---|---|---|---|---|
| Far (0–33%) | 27,289 | 1,265 | 0.8386 | 0.8578 | 0.8631 | 0.8635 |
| Mid (33–67%) | 85,657 | 3,763 | 0.7734 | 0.7753 | 0.7953 | 0.7931 |
| Near (67–100%) | 84,031 | 3,377 | 0.8065 | 0.8048 | 0.8095 | 0.8100 |

## RQ3 — market duration

AUC-ROC per bucket.

| Bucket | Snapshots | Markets | Market price | LR | GB | RF |
|---|---|---|---|---|---|---|
| ≤ 90d | 117,569 | 3,570 | 0.7648 | 0.7645 | 0.7675 | 0.7678 |
| 91–180d | 34,101 | 618 | 0.8391 | 0.8536 | 0.8670 | 0.8667 |
| 181–365d | 36,818 | 587 | 0.8658 | 0.8783 | 0.8831 | 0.8818 |
| > 365d | 8,489 | 149 | 0.7971 | 0.7863 | 0.7869 | 0.8024 |

## Time left before the scheduled close (v3 adds the last 14 days, ADR-022)

AUC-ROC per bucket.

| Bucket | Snapshots | Markets | Market price | LR | GB | RF |
|---|---|---|---|---|---|---|
| < 1 day | 2,743 | 1,464 | 0.7706 | 0.7602 | 0.7628 | 0.7635 |
| 1–7 days | 20,712 | 2,221 | 0.7738 | 0.7671 | 0.7706 | 0.7723 |
| 7–14 days | 32,737 | 2,994 | 0.7734 | 0.7702 | 0.7731 | 0.7738 |
| 14–30 days | 58,375 | 3,582 | 0.7804 | 0.7810 | 0.7801 | 0.7809 |
| ≥ 30 days | 82,410 | 1,979 | 0.8232 | 0.8364 | 0.8563 | 0.8539 |

## Trading simulation (paper §5.5)

Trade when |p_model − p_market| > 0.02; one token per trade; no fees or slippage. ROI = total profit / total capital; 95% CI from market-level bootstrap.

### Per-snapshot (paper's method)

| Strategy | Trades | Markets | Profit | ROI [95% CI] | Win rate |
|---|---|---|---|---|---|
| Always buy NO | 196,977 | 4,924 | $1,296 | 1.1% [-1.5%, 3.7%] | 61.6% |
| LR | 153,315 | 4,724 | $4,203 | 5.2% [2.3%, 8.2%] | 55.0% |
| GB | 153,884 | 4,700 | $7,640 | 8.5% [6.1%, 10.8%] | 63.6% |
| RF | 151,697 | 4,737 | $7,256 | 8.0% [5.5%, 10.4%] | 64.6% |

### One trade per market (first qualifying snapshot)

| Strategy | Trades | Markets | Profit | ROI [95% CI] | Win rate |
|---|---|---|---|---|---|
| Always buy NO | 4,924 | 4,924 | $118 | 3.7% [2.1%, 5.5%] | 67.4% |
| LR | 4,724 | 4,724 | $197 | 8.2% [5.8%, 10.6%] | 55.3% |
| GB | 4,700 | 4,700 | $187 | 6.4% [4.7%, 8.5%] | 65.8% |
| RF | 4,737 | 4,737 | $203 | 6.8% [5.0%, 8.5%] | 67.0% |
