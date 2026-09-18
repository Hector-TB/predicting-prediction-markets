# ADR-005: Leakage Fix — Removing Post-Resolution Snapshots

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/fix_leakage.py`

---

## Context

After building the initial snapshot dataset (~4.1M rows), we discovered that many snapshots occur *after* a market has effectively already resolved — the price had already crossed 0.95 or 0.05, meaning the outcome was a near-certainty. Including these "stale" snapshots in the dataset creates leakage: the model learns to predict from prices that already encode the answer.

The original 14-day cutoff before close (ADR-002) was insufficient — markets often resolve or converge well before their scheduled end date.

## Decision

Three-pass filter applied in `fix_leakage.py`, producing `_clean.parquet` files (originals preserved):

**Pass 1 — API closedTime:** For markets where the Gamma API reports a `closedTime`, drop all snapshots after that timestamp.

**Pass 2 — Price-based cutoff (all markets):** Find the first snapshot where `price_at_snapshot >= 0.95` or `<= 0.05`. Keep only snapshots up to that timestamp + 14 days (inactivity grace period). For markets where the price never crosses the threshold, use the last snapshot where the price changed by > 0.001 + 14 days.

**Pass 3 — Hard cap:** Drop any snapshots after 2026-05-01 (absolute ceiling to prevent future-dated data errors).

Originals are never modified. Output: `polymarket_ml_dataset_clean.parquet` (1,448,142 rows — ~65% of the original 4.1M).

## Rationale

The price-based approach is more principled than a fixed date cutoff: it removes rows where the outcome is already known from the price alone. The 14-day grace period avoids cutting too aggressively for markets with a brief price spike that reverses.

The `closedTime` pass handles markets where the API knows the exact resolution moment, which is more precise than price inference.

## Alternatives considered

- **Stricter price threshold (0.90)** — would remove more rows but potentially cut legitimate mid-market data
- **No grace period** — removing exactly from the first price crossing was too aggressive; prices can temporarily hit 0.95 and reverse
- **Fix the build_snapshots.py cutoff instead** — would require re-fetching all price history; the separate fix script is safer and reproducible

## Assumptions

1. A price above 0.95 or below 0.05 is a strong signal of effective resolution
2. The 14-day grace period is sufficient to include the final genuine trading activity without keeping stale observations
3. The 2026-05-01 hard cap is safe — all markets in the dataset resolved before this date

## Consequences

- **Dataset reduced from ~4.1M to ~1.4M rows** (65% reduction)
- The removed rows were precisely the "easy" late-stage snapshots — their removal makes evaluation metrics harder to beat but more realistic
- `polymarket_ml_dataset_clean.parquet` is the canonical dataset; the raw file is retained as a reference
- All models should use `_clean` files

## Related ADRs

- ADR-002: Snapshot window design (the original design that proved insufficient)
- ADR-004: Outcome threshold (0.95 is used here as the leakage threshold too)
