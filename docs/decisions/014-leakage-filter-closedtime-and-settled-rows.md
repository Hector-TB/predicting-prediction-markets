# ADR-014: Leakage Filter — Working closedTime Lookup + Per-Row Settled-Price Rule

**Date:** 2026-09-27  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data_collection_pipeline/fix_leakage.py` (`fetch_closed_times`, `filter_snapshots`)

---

## Context

An audit of `fix_leakage.py` during the ADR-013 full re-fetch found two problems with the ADR-005 filter.

1. **Pass 1 (closedTime) never ran.** The cache `data/market_closed_times.csv` held 210 markets with **zero** populated `closed_time` values, and the script never re-fetched while the file existed. The fetch itself used offset pagination with `limit=500`, which ADR-013 showed returns only 100 rows per page (skipping 400 of every 500) and is rejected past offset ~3,000. So every clean parquet to date was filtered by price alone.
2. **The 14-day grace period kept settled rows.** Pass 2 kept every snapshot up to 14 days after the price first crossed 0.95 / 0.05, whether or not the price came back. On the in-progress re-fetch CSV this left 36,208 rows (13.8% of kept rows) priced at ≥0.95 or ≤0.05. The pre-ADR-014 `polymarket_ml_dataset_clean.parquet` is 22.8% such rows. These are near-certain "easy" predictions, so they probably inflate test AUC.

The issue matters most for markets decided long before their scheduled `endDate`: 40.7% of the 45,143 markets closed more than 7 days early, with a median of 42 days early among those. Example: "Will every Republican Senator vote to confirm Pete Hegseth?" has `endDate` 2025-12-31 but `closedTime` 2025-01-25.

## Decision

1. **closedTime lookup by ID.** Query `GET /markets?condition_ids=<id>&…&closed=true&limit=100` in batches of 100 (`closed=true` is required: without it resolved markets return nothing). Retry with backoff and fail loudly after 8 attempts (same policy as ADR-013). The cache is incremental: populated entries are reused, missing or null ones are re-fetched every run. Verified 2026-09-27: 45,142 / 45,143 markets returned a `closedTime` in ~450 calls.
2. **Pass 2 becomes a per-row rule.** Drop a snapshot if **its own** `price_at_snapshot` is ≥ 0.95 or ≤ 0.05. The 14-day grace period is removed.
3. **The inactivity cutoff is a fallback only.** It applies only to markets without a `closedTime`: drop rows more than 14 days after the last price change.
4. The hard cap (tomorrow midnight UTC) is unchanged.

Pass order: closedTime → settled price → inactivity (no-closedTime markets only) → hard cap.

## Rationale

- **No look-ahead in row selection.** Every rule that decides whether a row survives uses only information available at that snapshot, except `closedTime`, which only removes rows after the market was already over. A "keep extreme rows only if the price later reverts" rule was considered and rejected: it selects rows based on the future, so kept extreme-price rows would come mostly from reversing markets. That biases the model on exactly those rows.
- **Train and serve match.** Live scoring can apply the same rule: markets currently priced ≥0.95 / ≤0.05 are treated as settled and not scored.
- **closedTime is exact.** It replaces the "last price change + 14 days" guess for almost every market.

## Alternatives considered

- **Keep the 14-day grace period** — leaves 14% or more of kept rows near-certain; rejected.
- **Conditional grace (keep only if the price reverts)** — look-ahead selection bias, see Rationale; rejected.
- **Drop everything after the first crossing** — also causal, but discards legitimate rows after a genuine reversal and has no live-scoring equivalent other than tracking each market's price history; rejected in favour of the simpler per-row rule.
- **Store closedTime in the meta CSV at fetch time** — the `fetch_cache/` checkpoints don't include it, so this would need another multi-hour full re-fetch. The batched ID lookup takes ~10 minutes. Could be revisited if `fetch_markets.py` is touched again.

## Assumptions

1. A price ≥ 0.95 or ≤ 0.05 means the market is effectively settled (same threshold as ADR-004 / ADR-005).
2. Gamma `closedTime` is when trading actually stopped. Spot-checked: the Hegseth and Bernhardt markets' price histories end on their `closedTime` dates.
3. `condition_ids` + `closed=true` keeps returning all requested markets. The fetch fails loudly on errors, and markets not returned fall back to the inactivity rule.

## Consequences

- **On the first 2,356 re-fetched markets (851,917 rows):** closedTime removes 138,674 rows and settled price 397,655. 315,588 rows remain, with 0 settled rows and 8,938 rows missing rolling features (later filled by `fix_dataset.py`). 424 markets lose all their rows.
- **Class balance shifts:** the row-level YES rate goes from 18.5% to 30.5%, because NO markets tend to sit near 0 for long stretches before closing. `class_weight='balanced'` / `scale_pos_weight` (ADR-007) must be recomputed from the new data, not hard-coded.
- **Metrics are not comparable** with any run on the pre-ADR-014 clean parquet. Expect lower AUC, which is the point. The market-price baseline (AUC 0.964 / log-loss 0.171) must also be recomputed on the new clean data: with near-certain rows gone, it is no longer the right bar.
- `end_date`, `duration_days`, `days_before_close` and `pct_lifetime_elapsed` stay based on the **scheduled** `endDate`. It is known in advance, so it is a legitimate feature. `closedTime` is future information and must never become a model feature.

## Related ADRs

- ADR-004: Outcome threshold (0.95 reused)
- ADR-005: Original leakage fix (Pass 1 and Pass 2 revised here)
- ADR-007: Class imbalance (weights must be recomputed)
- ADR-013: Keyset pagination (same offset-cap failure; same fail-loud retry policy)
