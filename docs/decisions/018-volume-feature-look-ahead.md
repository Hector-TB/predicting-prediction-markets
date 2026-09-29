# ADR-018: Volume Features Use Each Market's Final Lifetime Volume (Look-Ahead)

**Date:** 2026-09-27  
**Status:** Accepted — option 1 (drop the feature), 2026-09-28, as part of ADR-021  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data_collection_pipeline/fetch_markets.py` (`total_volume` ← Gamma `volumeNum`), `data_collection_pipeline/build_snapshots.py` (`total_volume`, `log_volume` per snapshot), all four `models/*/train.py` (`log_volume` feature)

---

## Context

`fetch_markets.py` fetches **closed** markets and stores Gamma's `volumeNum`: the market's total trading volume over its whole life, known only once it closes. `build_snapshots.py` then copies that single number onto **every** snapshot of the market as `total_volume` and `log_volume`. LR, XGBoost, RF and RF + Trends all use `log_volume` as a feature (it is 5th by importance in the paper's XGBoost model).

So a snapshot taken in a market's first week "knows" how much money the market will eventually attract. That is information from the future, and it may be related to the outcome: markets that attract late volume are often ones where something happened. The size of the effect is unmeasured.

It also breaks live prediction (ADR-017): an open market only has volume **to date**, so the feature the models were trained on doesn't exist at prediction time.

A milder form affects market selection: the ADR-008 filter (volume ≥ $1k) is also applied to final volume, so markets enter the dataset partly because of what happened after each snapshot.

## Options

1. **Drop `total_volume` / `log_volume` from the models.** Simple, removes the look-ahead completely, and live prediction needs nothing extra. Loses whatever genuine signal volume carries (liquidity, attention).
2. **Volume-to-date at each snapshot.** Rebuild the feature from trade history, as cumulative volume up to the snapshot time (e.g. Polymarket's data API trades by market). Keeps the signal honestly, but needs a new fetch over ~45k markets (paginated trade histories) and pipeline changes; feasibility and cost not yet checked.
3. **Keep final volume.** Rejected: the look-ahead would stay, and live prediction would have to feed a different quantity than the models were trained on.

## Recommendation (pending decision)

Do **option 1 first**: retrain v2 without volume features, and measure the change against the with-volume models using `rescore_paper.py` (ADR-015). That tells us how much the models relied on future volume. If volume looks genuinely useful, add **option 2** later as its own ADR and dataset version.

For the selection effect: keep the $1k filter for training (changing it changes the dataset). For live prediction, apply it to volume-to-date and warn below $1k (ADR-017).

## Decision (2026-09-28)

Option 1: `log_volume` is removed from LR, XGBoost, RF, RF + Trends, SVM and the smoke test. The columns stay in the dataset, but no model uses them. We skipped the with/without comparison: the feature isn't available at prediction time either way.

**Option 2 feasibility (checked 2026-09-28):** `data-api.polymarket.com/trades?market=<conditionId>` returns timestamped trades (sample: $36,373 summed vs $36,884 `volumeNum`). But pagination stops at offset 10,000 (about 10.5k trades), so large markets are truncated. Full history for those would need another source (e.g. the Polymarket orderbook subgraph). Tracked as a separate task in ROADMAP.

## Consequences

- v2 results are quoted from the no-volume models. Comparisons with the paper (which used final volume) note the difference.
- `rescore_paper.py` gains a with/without-volume comparison for the record.
- The live `/predict` path needs no volume history.

## Related ADRs

- ADR-008: Market filters (volume ≥ $1k applied to final volume)
- ADR-015: Evaluation methodology (how the with/without comparison is measured)
- ADR-017: On-demand prediction (needs features that exist at prediction time)
