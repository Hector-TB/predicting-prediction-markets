# Course paper

`paper.tex` is the final DS-GA 1003 paper (LaTeX source only; the `figures/` it references are not in the repo).

## How its results relate to the current pipeline (as of 2026-09-27)

- **Headline (RQ1):** market baseline AUC 0.8890 / log-loss 0.3278 vs RF + Trends 0.9034 / 0.3011 on 288,490 test snapshots. `scripts/print_metrics.py` reproduces the baseline on the old `polymarket_ml_dataset_clean.parquet` (0.8905; 0.8919 on RF's test markets). The README's older 0.964 figure came from the unfiltered parquet and was wrong.
- **Section 3.3 overstates the leakage filtering.** It says snapshots after the API `closedTime` were dropped. In the code that pass never ran: the cache held zero close times (ADR-014). The paper's data was filtered by price only, with a 14-day grace period, so its test set includes post-close snapshots with carried-forward prices and about 22.5% near-settled rows.
- **Trading simulation (5.5)** treats every snapshot as a separate trade (~52 per market), and some trades use stale post-close prices. Treat it as an upper bound, even beyond the caveats the paper lists.
- All paper numbers need re-measuring on the ADR-013/014 dataset before being quoted again.
