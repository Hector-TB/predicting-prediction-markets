# Course paper

`paper.tex` is the final DS-GA 1003 paper (LaTeX source only; the `figures/` it references are not in the repo).

## How its results relate to the current pipeline (as of 2026-09-27)

- **Headline (RQ1):** market baseline AUC 0.8890 / log-loss 0.3278 vs RF + Trends 0.9034 / 0.3011 on 288,490 test snapshots. `scripts/print_metrics.py` reproduces the baseline on the old `polymarket_ml_dataset_clean.parquet` (0.8905; 0.8919 on RF's test markets). The README's older 0.964 figure came from the unfiltered parquet and was wrong.
- **Section 3.3 overstates the leakage filtering.** It says snapshots after the API `closedTime` were dropped. In the code that pass never ran: the cache held zero close times (ADR-014). The paper's data was filtered by price only, with a 14-day grace period, so its test set includes post-close snapshots with carried-forward prices and about 22.5% near-settled rows.
- **Trading simulation (5.5)** treats every snapshot as a separate trade (~52 per market), and some trades use stale post-close prices. Treat it as an upper bound, even beyond the caveats the paper lists.
- All paper numbers need re-measuring on the ADR-013/014 dataset before being quoted again.

## Reproduction check (2026-09-27)

`analysis/rescore_paper.py` run on the course-era data reproduces the paper's RQ1 table, duration breakdown and trade counts/profit exactly ([`rescore_course_data.md`](rescore_course_data.md)). New findings:

- **The RQ1 gains are significant.** Market-clustered bootstrap 95% CIs on ΔAUC exclude zero for every tree model (RF + Trends +0.0143 [+0.0054, +0.0231]). LR's include zero.
- **The paper's ROI figures are overstated.** `analysis/trading_simulation.ipynb` counts capital as the market price for every trade, including NO bets, which actually cost 1 − price. With the correct cost, GB + Trends earns **12.7%** ROI per snapshot (paper: 27.5%) and **11.2%** with one trade per market. That's still above always-buy-NO (7.3% / 5.7%). The horizon sweep ("48% at day 11") uses the same formula.
- **The lifecycle figure (RQ3) used thirds of each market's snapshots by order** (`analysis.ipynb` cell 21), not `pct_lifetime_elapsed` as §5.3's text says. With snapshot-order thirds the script reproduces the figure closely (Far: trees +0.017 to +0.020 over the market; Near RF 0.939 vs the paper's 0.937). With `pct_lifetime_elapsed` the numbers differ (Near RF 0.903). `rescore_course_data.md` predates this and shows only the `pct_lifetime_elapsed` version.
