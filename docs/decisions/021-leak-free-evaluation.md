# ADR-021: Leak-Free Evaluation — Split by Resolution Time, No Test-Set Tuning

**Date:** 2026-09-28  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Supersedes:** ADR-001 (split rule only; market-level splitting and the 80/20 ratio stay)  
**Code location:** `scripts/recompute_split.py`, `data_collection_pipeline/fix_leakage.py` (`split_cutoff`, `mark_pre_cutoff`), `data_collection_pipeline/fetch_markets.py` (new markets → test), `models/*/train.py` (thresholds, LR CV)

---

## Context

A review before the first v2 run found four ways information from the future, or from the test set itself, reached the test scores:

1. **The split was on `start_date` (ADR-001).** A market's label is only known once it resolves. 7,041 of 36,114 train markets (19%) started before the first test market but resolved after it. So the model trained on outcomes that weren't known yet at the time of many test snapshots. This matters most for related markets (the same election, sports season or asset price ladder).
2. **Decision thresholds were chosen on the test set.** Every model picked its YES/NO cut-off to maximise F1 on the test predictions. AUC and log-loss don't use a threshold, so they were unaffected. But F1, accuracy, `pred_label` and anything built on them were optimistic.
3. **Logistic regression tuned with ungrouped 5-fold CV.** Snapshots of one market landed in both the fitting and the validation fold, so the grid search tuned on near-duplicate rows. This is not a test-set leak, but it gives optimistic CV scores and can pick the wrong regularisation.
4. **`log_volume` is the final lifetime volume** (ADR-018).

## Decision

1. **Split by resolution time, at one cutoff T.**
   - Sort markets by `closedTime` (scheduled `end_date` if missing); the first 80% are **train**, the rest **test**. T is the latest train resolution.
   - `fix_leakage.py` relabels test rows dated before T as `split = "test_pre_cutoff"`. Models only read `train` and `test`, so these rows are unused.
   - Result: every training label was known by T, and every test snapshot is dated T or later. This mirrors real use: on date T, train on everything settled, then predict what is still open.
   - Newly fetched markets are always assigned `test`. That is safe even for a market that resolved before T: all its rows are dated before T, so they are all dropped.
2. **Thresholds come from held-out training data.** RF, RF + Trends and XGBoost use the calibration markets they already hold out. LR uses out-of-fold predictions on train (grouped by market).
3. **LR grid search uses `StratifiedGroupKFold` grouped by market**, like RF.
4. **Drop `log_volume`** from every model (accepts ADR-018 option 1).

## Rationale

The project's central question is whether engineered features beat the market price. The market price never has any of these advantages, so each leak biases the comparison towards the models. The resolution-time split is the only simple rule under which no training label postdates any test snapshot.

## Alternatives considered

- **Keep the start-date split and drop overlapping train markets** (train markets resolved after the first test start). This fixes the train side, but keeps test snapshots from before the last train resolution. It also leaves the cutoff defined by two different dates. Rejected in favour of one date T.
- **Split by resolution time without dropping pre-T test rows.** 65% of test rows would be dated before T, which is the same leak in another form. Rejected.
- **Walk-forward (several cutoffs).** More robust estimates, but a bigger change to every training script. Worth revisiting once the one-command refresh exists (ADR-019).

## Assumptions

1. `closedTime` is when the outcome became known. It is 45,142 of 45,143 populated.
2. Test markets are those resolved by the data-collection date. Markets still open are left out, because they have no label. This is a **selection bias** (test leans towards shorter markets), not a leak. Report it alongside the results.

## Consequences

- On v2 (at adoption): T = 2026-07-01 07:11 UTC; 36,114 train and 9,029 test markets, and 4,974 markets changed side. Before the leakage filter, an estimated ~135k test rows remain after the pre-T drop, similar to the old split's test size. Train loses about 12% of rows.
- **Results are not comparable** with any run on the start-date split or with `log_volume`. The market baseline must be re-scored on the same test rows (ADR-015).
- The split must be recomputed with `scripts/recompute_split.py` after a full re-fetch, then the pipeline re-run.
- `train.py` scripts must never select rows with `split != "train"` for fitting, or `split != "test"` for scoring.

## Related ADRs

- ADR-001: Temporal market-level split (superseded here for the split rule)
- ADR-013: Full re-fetch → recompute split
- ADR-014: Leakage filter (same script now applies the pre-cutoff rule)
- ADR-015: Evaluation methodology (shared test rows, market-level bootstrap)
- ADR-018: Volume look-ahead (option 1 adopted here)
