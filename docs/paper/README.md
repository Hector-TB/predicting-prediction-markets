# Course paper

`paper.tex` is the final DS-GA 1003 paper (LaTeX source only; the `figures/` it references are not in the repo).

## How its results relate to the current pipeline (as of 2026-09-27)

- **Headline (RQ1):** market baseline AUC 0.8890 / log-loss 0.3278 vs RF + Trends 0.9034 / 0.3011 on 288,490 test snapshots. `scripts/print_metrics.py` reproduces the baseline on the old `polymarket_ml_dataset_clean.parquet` (0.8905; 0.8919 on RF's test markets). The README's older 0.964 figure came from the unfiltered parquet and was wrong.
- **Section 3.3 overstates the leakage filtering.** It says snapshots after the API `closedTime` were dropped. In the code that pass never ran: the cache held zero close times (ADR-014). The paper's data was filtered by price only, with a 14-day grace period, so its test set includes post-close snapshots with carried-forward prices and about 22.5% near-settled rows.
- **Trading simulation (5.5)** treats every snapshot as a separate trade (~52 per market), and some trades use stale post-close prices. Treat it as an upper bound, even beyond the caveats the paper lists.
- All paper numbers needed re-measuring on the leak-free data before being quoted again. RQ1, RQ3 and the trading simulation now are (v3, below); RQ2 (Google Trends) waits for the Trends rework.

## Reproduction check (2026-09-27)

`analysis/rescore_paper.py` run on the v1 dataset reproduces the paper's RQ1 table, duration breakdown and trade counts/profit exactly ([`rescore_v1_data.md`](rescore_v1_data.md)). New findings:

- **The RQ1 gains are significant.** Market-clustered bootstrap 95% CIs on ΔAUC exclude zero for every tree model (RF + Trends +0.0143 [+0.0054, +0.0231]). LR's include zero.
- **The paper's ROI figures are overstated.** `analysis/trading_simulation.ipynb` counts capital as the market price for every trade, including NO bets, which actually cost 1 − price. With the correct cost, GB + Trends earns **12.7%** ROI per snapshot (paper: 27.5%) and **11.2%** with one trade per market. That's still above always-buy-NO (7.3% / 5.7%). The horizon sweep ("48% at day 11") uses the same formula.
- **The lifecycle figure (RQ3) used thirds of each market's snapshots by order** (`analysis.ipynb` cell 21), not `pct_lifetime_elapsed` as §5.3's text says. With snapshot-order thirds the script reproduces the figure closely (Far: trees +0.017 to +0.020 over the market; Near RF 0.939 vs the paper's 0.937). With `pct_lifetime_elapsed` the numbers differ (Near RF 0.903). `rescore_v1_data.md` predates this and shows only the `pct_lifetime_elapsed` version.

## Re-score on v3 (2026-09-29)

[`rescore_v3_data.md`](rescore_v3_data.md): LR, XGBoost and RF retrained on dataset v3. The data has a working leakage filter and no 14-day pre-close cutoff, and the models use the split by resolution time and the shared training protocol (ADR-014, ADR-021, ADR-022, ADR-024). The test set has 196,977 snapshots across 4,924 markets. Trends variants were not trained.

- **RQ1 holds.** XGBoost +0.0141 AUC over the market [+0.0092, +0.0203], RF +0.0136 [+0.0090, +0.0194]. Both also have lower log-loss and Brier scores than the market. LR +0.0029 [−0.0012, +0.0069], level with the market, as in the paper.
- **Absolute levels are lower than the paper's** (market AUC 0.799 vs 0.889). The leakage filter removes near-settled and post-close rows, which were easy to predict. So only differences from the market on the same rows are comparable across versions, not raw AUCs.
- **RQ3 holds and is sharper.** By snapshot-order thirds, the gain is largest in the first third (+0.022 AUC) and smallest in the last (+0.009). By time left before the scheduled close, all of the gain is ≥ 30 days out (XGBoost 0.856 vs market 0.823). In the last 30 days the models match the market, and in the last week they are slightly behind it. This bucket is confounded with duration: only longer markets have snapshots ≥ 30 days out.
- **Trading (NO bets costed at 1 − p, ADR-015; no fees or spread):** per snapshot, XGBoost 8.5% [6.1%, 10.8%] and RF 8.0% vs always-NO 1.1%. With one trade per market, 6.4–8.2% vs 3.7%.
- **Base rate drift:** 25.4% of training snapshots resolve YES vs 38.4% in test, rising steadily over time. It is handled by calibrating on the newest training markets (ADR-024) and is visible in the log-loss.
