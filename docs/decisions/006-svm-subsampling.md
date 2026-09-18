# ADR-006: SVM Fixed-Percentile Subsampling (7 snapshots per market)

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `models/svm/svm.py:22–99`

---

## Context

SVMs have O(n²) to O(n³) time complexity in the number of training samples. Training on 1.4M rows with RBF kernel is computationally infeasible. We needed a principled way to reduce the dataset for SVM training without destroying the statistical structure.

Naive random subsampling would give different numbers of snapshots per market (because markets have different lengths), meaning long markets dominate training.

## Decision

For each market, select exactly **7 snapshots** corresponding to fixed percentiles of the market's lifetime: 20%, 30%, 40%, 50%, 60%, 70%, 80%. For each percentile, find the snapshot whose `pct_lifetime_elapsed` is closest to the target — never interpolate.

This reduces the dataset from ~1.4M to ~147k rows (7 per market × ~21k markets).

## Rationale

- **Equal market contribution:** every market contributes exactly 7 rows regardless of duration; long markets don't dominate
- **Consistent lifecycle coverage:** the 7 percentile points give a standardized view of the market at early, middle, and late stages
- **Computationally tractable:** ~147k rows makes RBF-SVM feasible in minutes rather than days
- **No fabricated data:** we pick the closest real snapshot, never interpolate

The percentile 20%–80% range avoids very early (burn-in period) and very late (near resolution) snapshots by design.

## Alternatives considered

- **Random subsample** — would give unequal market representation; rejected
- **One snapshot per market** — too little lifecycle information; rejected
- **Uniform time intervals (e.g. every 30 days)** — markets have different lengths, so 30-day intervals give different coverage for different markets; percentiles are cleaner
- **Use LightGBM instead of SVM** — tree models can train on full 1.4M rows; SVM was chosen as one of the required course models

## Assumptions

1. 7 lifecycle snapshots per market are sufficient to learn the prediction task
2. The 20%–80% range captures the most informative lifecycle window
3. Percentile-based subsampling preserves the statistical properties of the full dataset

## Consequences

- SVM cannot be directly compared to LR/XGBoost on the same test set rows (different snapshot selection)
- The `target_percentile` column is added as a feature — it replaces `pct_lifetime_elapsed` (r = 0.93 with it, so it's cleaner)
- SVM predictions are per-market (7 predictions), not per-snapshot
- This subsampling only applies to SVM; LR and XGBoost train on the full dataset

## Related ADRs

- ADR-002: Snapshot window design
- ADR-001: Train/test split (still applied at market level before subsampling)
