# ADR-015: Evaluation Methodology — Shared Test Rows, Market-Level CIs, Corrected Trading Simulation

**Date:** 2026-09-27  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Code location:** `analysis/rescore_paper.py`, `models/common/evaluation.py` (`compute_metrics`, `bootstrap_auc_diff(groups=...)`), `scripts/print_metrics.py`

---

## Context

The project's central question (RQ1) is whether models beat the market price. Re-checking the course paper's evaluation on 2026-09-27 found several problems with how that question had been answered:

1. **The baseline was quoted from the wrong data.** The README and CLAUDE.md quoted the market baseline as AUC 0.964 / log-loss 0.171. That figure came from the unfiltered parquet; on the leakage-filtered data the models were scored on, it is 0.8890 / 0.3278. The README therefore concluded "no model beats the market", which contradicted the paper.
2. **Models and baseline were scored on different row sets.** `print_metrics.py` scored the baseline on all 299,576 clean test snapshots and the models on the 288,490 they predicted.
3. **The confidence intervals were too narrow.** `bootstrap_auc_diff` resampled snapshots independently, but each test market contributes about 70 strongly correlated snapshots.
4. **The trading ROI was overstated.** `analysis/trading_simulation.ipynb` counted capital as the market price `p` for every trade. A NO token costs `1 − p`, so capital was understated and GB + Trends ROI was reported as 27.5% instead of 12.7%.
5. **The per-snapshot trading simulation counts correlated repeat bets.** It treats every snapshot as an independent trade (~52 per market).
6. **The lifecycle definition was ambiguous.** The paper's text describes thirds of `pct_lifetime_elapsed`, but its figure used thirds of each market's snapshots by order (`analysis.ipynb` cell 21); the two give different numbers.

## Decision

1. **Baseline = market price as a probability** (`price_at_snapshot`), scored with the same metrics as the models. It is always computed from the data, never hard-coded in docs or code. (The paper's §4.1 describes a 0.5-threshold classifier, but the reported AUCs use the price as a probability.)
2. **Every comparison uses one shared evaluation set:** the inner join of the clean test snapshots with every available model's predictions on `(market_id, snapshot_timestamp)`. Prediction CSVs must therefore include `snapshot_timestamp` (already a rule in `.claude/rules/models.md`; LR/GB now comply).
3. **Uncertainty is estimated by resampling markets, not snapshots.** `bootstrap_auc_diff(..., groups=market_id)` is the default for model-vs-market comparisons; trading ROI CIs also resample markets.
4. **Trading simulation:** a YES trade costs `p` and a NO trade costs `1 − p`; ROI = total profit / total capital. **One trade per market** (the first snapshot where |p_model − p_market| > 0.02) is the primary result. The per-snapshot version is kept only for comparison with the paper. Both are reported against an always-buy-NO baseline, with no fees or slippage.
5. **Lifecycle (RQ3)** is reported both ways, labelled: by snapshot order within each market (the paper's figure) and by `pct_lifetime_elapsed`.

## Rationale

- One shared row set is the only way to say "model X beats the market" without the comparison being partly about which rows were scored.
- Resampling markets matches the unit of independence. Results that only look significant under row resampling are not significant.
- Pricing NO bets at `1 − p` is what the trade actually costs; any other capital definition makes ROI meaningless.
- One trade per market removes the dependence between repeated bets on the same outcome and is closer to what a real strategy could execute.

## Alternatives considered

- **Keep per-snapshot trading as primary** — comparable with the paper, but its trade count and profit scale with how many snapshots a market has, not with how many independent decisions were made; kept as secondary.
- **Row-level bootstrap** — cheaper, but understates uncertainty for correlated snapshots; rejected.
- **Pick one lifecycle definition** — would silently change RQ3's numbers relative to the paper; report both until the paper is revised.

## Assumptions

1. Markets are independent of each other. Related markets (e.g. several candidates in one election) are not; if that dependence turns out to matter, resample events instead of markets.
2. Buying one token at the snapshot price was possible at that moment. Liquidity, spread and fees are not modelled, so all trading results are upper bounds.

## Consequences

- **Validated on the v1 dataset:** `rescore_paper.py` reproduces the paper's RQ1 table, duration breakdown and trade counts exactly, and its lifecycle figure closely with snapshot-order thirds (Near RF 0.939 vs the paper's 0.937) ([`docs/paper/rescore_v1_data.md`](../paper/rescore_v1_data.md)).
- **RQ1 holds with honest CIs on that data:** every tree model's ΔAUC CI excludes zero (e.g. RF + Trends +0.0143 [+0.0054, +0.0231]); LR's does not.
- **The paper's trading ROI claims (27.5%, 48% at day 11) are overstated** and should not be quoted. Corrected: 12.7% per snapshot, 11.2% one per market, vs 7.3% / 5.7% for always-buy-NO.
- All results must be re-measured on the ADR-013/014 dataset; this ADR fixes how, not what the numbers will be.

## Amendment (2026-09-29): more comparisons in the re-score

`analysis/rescore_paper.py` now also reports:

1. **Market-level CIs on the log-loss and Brier differences** (`bootstrap_loss_diff`), not just on ΔAUC. Log-loss is a primary metric, so its gap needs an interval too.
2. **Every comparison with each market counted once**: snapshot weight 1 / its market's snapshot count, matching training (ADR-024). The row-weighted numbers stay the headline; the weighted ones show whether the result rests on a few long markets.
3. **Time left before close, split by market duration.** Only longer markets have snapshots ≥ 30 days out, so the plain time-left table confounds the two.
4. **Trading over thresholds (τ = 0.02, 0.05, 0.10) and costs per token (0, 1, 2 cents).** YES costs p + c, NO costs 1 − p + c. Fills at the recorded price remain an assumption: we have no order-book data.

## Related ADRs

- ADR-001: Temporal train/test split (defines the test set)
- ADR-007: Class imbalance (why AUC / log-loss, not accuracy)
- ADR-014: Leakage filter (the dataset these rules will be applied to next)
