# Paper re-score — polymarket_ml_dataset_clean.parquet

## RQ1 — models vs the market price

196,977 test snapshots, 4,924 markets. Every snapshot counted.

| Model | AUC-ROC | PR-AUC | Log-loss | Brier |
|---|---|---|---|---|
| Market price | 0.7986 | 0.7390 | 0.5171 | 0.1727 |
| LR | 0.8014 | 0.7281 | 0.5223 | 0.1730 |
| GB | 0.8127 | 0.7414 | 0.5105 | 0.1692 |
| RF | 0.8122 | 0.7374 | 0.5097 | 0.1692 |

Differences vs the market (model − market), 95% CI from 500 market-level bootstrap resamples. Higher AUC is better; **lower (negative) log-loss and Brier are better**.

| Model | ΔAUC [95% CI] | ΔLog-loss [95% CI] | ΔBrier [95% CI] |
|---|---|---|---|
| LR | +0.0029 [-0.0012, +0.0069] | +0.0052 [-0.0014, +0.0123] | +0.0003 [-0.0017, +0.0024] |
| GB | +0.0141 [+0.0092, +0.0203] | -0.0066 [-0.0142, +0.0008] | -0.0035 [-0.0062, -0.0008] |
| RF | +0.0136 [+0.0090, +0.0194] | -0.0074 [-0.0143, -0.0006] | -0.0035 [-0.0059, -0.0011] |

## RQ1 — each market counted once

Same snapshots, each weighted by 1 / its market's snapshot count, so every market counts once (as in training, ADR-024). Shows whether the result holds across typical markets or rests on long ones.

| Model | AUC-ROC | PR-AUC | Log-loss | Brier |
|---|---|---|---|---|
| Market price | 0.8595 | 0.7806 | 0.4268 | 0.1369 |
| LR | 0.8636 | 0.7726 | 0.4254 | 0.1356 |
| GB | 0.8706 | 0.7827 | 0.4137 | 0.1324 |
| RF | 0.8707 | 0.7796 | 0.4131 | 0.1324 |

Differences vs the market (model − market), 95% CI from 500 market-level bootstrap resamples. Higher AUC is better; **lower (negative) log-loss and Brier are better**.

| Model | ΔAUC [95% CI] | ΔLog-loss [95% CI] | ΔBrier [95% CI] |
|---|---|---|---|
| LR | +0.0041 [+0.0014, +0.0068] | -0.0014 [-0.0055, +0.0030] | -0.0013 [-0.0029, +0.0002] |
| GB | +0.0111 [+0.0078, +0.0143] | -0.0131 [-0.0183, -0.0075] | -0.0045 [-0.0060, -0.0029] |
| RF | +0.0112 [+0.0081, +0.0144] | -0.0137 [-0.0185, -0.0082] | -0.0045 [-0.0061, -0.0029] |

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

## Time left × market duration

Only markets longer than 30 days can have snapshots ≥ 30 days before close, so the time-left table mixes "early" with "long market". Here it is split by duration. Cells: market AUC, then each model's AUC minus the market's. Buckets with few markets are noisy.

### Duration ≤ 90d

| Time left | Snapshots | Markets | Market AUC | Δ LR | Δ GB | Δ RF |
|---|---|---|---|---|---|---|
| < 1 day | 2,370 | 1,269 | 0.7649 | -0.0119 | -0.0108 | -0.0090 |
| 1–7 days | 18,015 | 1,937 | 0.7670 | -0.0080 | -0.0040 | -0.0016 |
| 7–14 days | 29,009 | 2,656 | 0.7639 | -0.0056 | -0.0023 | -0.0012 |
| 14–30 days | 48,176 | 3,117 | 0.7574 | -0.0012 | -0.0020 | -0.0013 |
| ≥ 30 days | 19,999 | 752 | 0.7813 | +0.0117 | +0.0241 | +0.0216 |

### Duration 91–365d

| Time left | Snapshots | Markets | Market AUC | Δ LR | Δ GB | Δ RF |
|---|---|---|---|---|---|---|
| < 1 day | 369 | 193 | 0.7998 | -0.0006 | +0.0028 | -0.0029 |
| 1–7 days | 2,663 | 280 | 0.8003 | +0.0090 | +0.0023 | +0.0023 |
| 7–14 days | 3,660 | 332 | 0.8373 | +0.0112 | +0.0103 | +0.0080 |
| 14–30 days | 10,145 | 459 | 0.8703 | +0.0030 | +0.0046 | +0.0054 |
| ≥ 30 days | 54,082 | 1,085 | 0.8504 | +0.0188 | +0.0334 | +0.0319 |

### Duration > 365d

| Time left | Snapshots | Markets | Market AUC | Δ LR | Δ GB | Δ RF |
|---|---|---|---|---|---|---|
| < 1 day | 4 | 2 | 1.0000 | +0.0000 | +0.0000 | +0.0000 |
| 1–7 days | 34 | 4 | 0.7576 | +0.0606 | +0.0379 | -0.0076 |
| 7–14 days | 68 | 6 | 0.5701 | -0.0728 | -0.0331 | -0.0985 |
| 14–30 days | 54 | 6 | 0.6000 | -0.0000 | -0.0667 | -0.0222 |
| ≥ 30 days | 8,329 | 142 | 0.7981 | -0.0137 | -0.0117 | +0.0045 |

## Trading simulation (paper §5.5)

Paper's rule: trade when |p_model − p_market| > 0.02; one token per trade; no fees or slippage. ROI = total profit / total capital; 95% CI from market-level bootstrap.

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

## Trading with costs and other thresholds

Each token costs its price plus c (spread and fees): YES costs p + c, NO costs 1 − p + c. τ = minimum gap between model and market to trade. Cells: ROI [95% CI]; trade counts don't depend on c.

### Per snapshot

| Strategy | τ | Trades | c = 0.00 | c = 0.01 | c = 0.02 |
|---|---|---|---|---|---|
| Always buy NO | — | 196,977 | 1.1% [-1.5%, 3.7%] | -0.6% [-3.0%, 2.0%] | -2.1% [-4.6%, 0.4%] |
| LR | 0.02 | 153,315 | 5.2% [2.3%, 8.2%] | 3.3% [0.4%, 6.1%] | 1.4% [-1.4%, 4.1%] |
| LR | 0.05 | 93,718 | 7.2% [3.6%, 10.5%] | 5.3% [1.7%, 8.5%] | 3.4% [-0.1%, 6.5%] |
| LR | 0.10 | 33,941 | 18.0% [13.4%, 22.6%] | 15.7% [11.2%, 20.2%] | 13.5% [9.0%, 17.9%] |
| GB | 0.02 | 153,884 | 8.5% [6.1%, 10.8%] | 6.6% [4.3%, 9.0%] | 4.9% [2.6%, 7.2%] |
| GB | 0.05 | 99,134 | 12.9% [9.9%, 16.1%] | 10.9% [8.0%, 14.1%] | 9.0% [6.2%, 12.2%] |
| GB | 0.10 | 51,604 | 20.3% [15.8%, 25.3%] | 18.1% [13.6%, 22.9%] | 15.9% [11.5%, 20.7%] |
| RF | 0.02 | 151,697 | 8.0% [5.5%, 10.4%] | 6.2% [3.8%, 8.6%] | 4.5% [2.1%, 6.8%] |
| RF | 0.05 | 97,813 | 12.8% [9.5%, 16.1%] | 10.8% [7.6%, 14.0%] | 8.9% [5.7%, 12.0%] |
| RF | 0.10 | 47,886 | 19.5% [14.6%, 24.9%] | 17.3% [12.4%, 22.5%] | 15.1% [10.3%, 20.2%] |

### One trade per market

| Strategy | τ | Trades | c = 0.00 | c = 0.01 | c = 0.02 |
|---|---|---|---|---|---|
| Always buy NO | — | 4,924 | 3.7% [2.1%, 5.5%] | 2.1% [0.5%, 3.9%] | 0.6% [-1.0%, 2.3%] |
| LR | 0.02 | 4,724 | 8.2% [5.8%, 10.6%] | 6.1% [3.7%, 8.5%] | 4.1% [1.8%, 6.5%] |
| LR | 0.05 | 4,093 | 9.7% [7.3%, 12.0%] | 7.7% [5.3%, 10.0%] | 5.7% [3.4%, 7.9%] |
| LR | 0.10 | 2,758 | 16.4% [13.3%, 19.8%] | 14.1% [11.1%, 17.5%] | 12.0% [9.0%, 15.3%] |
| GB | 0.02 | 4,700 | 6.4% [4.7%, 8.5%] | 4.7% [3.0%, 6.8%] | 3.1% [1.4%, 5.1%] |
| GB | 0.05 | 4,164 | 9.4% [7.1%, 11.1%] | 7.6% [5.3%, 9.3%] | 5.8% [3.7%, 7.6%] |
| GB | 0.10 | 3,061 | 16.5% [13.7%, 19.2%] | 14.3% [11.6%, 17.0%] | 12.3% [9.6%, 14.9%] |
| RF | 0.02 | 4,737 | 6.8% [5.0%, 8.5%] | 5.1% [3.4%, 6.8%] | 3.5% [1.8%, 5.1%] |
| RF | 0.05 | 4,200 | 9.4% [7.3%, 11.6%] | 7.6% [5.5%, 9.7%] | 5.8% [3.8%, 8.0%] |
| RF | 0.10 | 3,000 | 16.7% [13.9%, 19.9%] | 14.5% [11.8%, 17.7%] | 12.5% [9.8%, 15.6%] |
