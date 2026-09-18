# ADR-001: Temporal, Market-Level 80/20 Train/Test Split

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/fetch_markets.py:271–276`

---

## Context

The dataset has ~1.4M rows but only ~21k distinct markets. Each market generates ~70 snapshots on average, so naive row-level splitting would leak future snapshots of a training market into the test set.

## Decision

Split at the **market level**, using a **temporal cutoff**: the oldest 80% of markets by `start_date` form the train set; the newest 20% form the test set.

- Train: 16,774 markets / 1,159,652 snapshots
- Test: 4,174 markets / 288,490 snapshots
- Cutoff date is determined at runtime by percentile of `start_date`

## Rationale

- **Market-level** ensures no snapshots from the same market appear in both splits — prevents direct leakage
- **Temporal** order mimics the real-world deployment scenario: train on older markets, predict on newer ones; random splits would be over-optimistic
- 80/20 is standard; the large number of markets means even 20% gives 4k+ markets for evaluation

## Alternatives considered

- **Random market-level split** — simpler but not temporally honest; newer markets may have different dynamics (Polymarket grew significantly post-2023)
- **5-fold cross-validation by market** — would give better variance estimates but was too expensive computationally given snapshot count
- **Row-level random split** — rejected immediately; direct leakage of market-level outcome labels

## Assumptions

1. Temporal order of `start_date` is a reasonable proxy for "old vs new market" — true given Polymarket's growth trajectory
2. Market dynamics are not severely non-stationary across the split (if 2024–2025 markets behave fundamentally differently from 2023 markets, the split is still correct but performance may appear worse than it is)

## Consequences

- **Easier:** evaluation reflects realistic deployment; no data leakage
- **Harder:** train/test class balance is not guaranteed (slight difference: train 22.4% YES vs test 21.0% YES); this is acceptable
- **Risk:** if the market is currently in a regime shift, historical performance may understate or overstate future performance

## Related ADRs

- ADR-008: Market filters (which markets are eligible for the split)
