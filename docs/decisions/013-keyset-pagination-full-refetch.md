# ADR-013: Keyset Pagination for Market Fetch + Full Re-fetch

**Date:** 2026-09-25  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data_collection_pipeline/fetch_markets.py` (`_fetch_window`, `main --full`)

---

## Context

On 2026-09-25 the incremental fetch (June → September 2026) was found to be silently dropping markets:

1. **Page size cap.** The Gamma `/markets` endpoint returns at most 100 rows per request, even with `limit=500`. The script advanced `offset` by 500 each page, skipping 400 of every 500 markets.
2. **Offset cap.** `/markets` rejects offsets beyond ~3,000 (`422: offset too large, use /markets/keyset for deeper pagination`). Since mid-2026 a single day can hold 3,000+ markets with ≥$1k volume (mostly 5-minute / hourly crypto "Up or Down" and sports O/U markets), so window-halving bottoms out at one day and whole days were skipped.
3. **Silent skips.** Pages that failed after all retries were skipped with a log line, leaving gaps. Because the incremental fetch resumes from the newest `start_date`, those gaps would never be revisited.

A spot check of 2025-03-01 → 2025-03-15 with keyset pagination found 386 markets passing the ADR-008 filters; the stored meta CSV had 324 (84%). Historical data is therefore also incomplete, though the exact cause of each missing market is unconfirmed.

## Decision

1. Replace offset pagination with the cursor-based `/markets/keyset` endpoint (`limit=100`, `after_cursor=<next_cursor>`). No offset cap, so window splitting is removed. Monthly windows are kept as checkpoint units.
2. **Fail loudly:** a page that fails after all retries aborts the fetch instead of being skipped.
3. Parse and filter each window as it arrives (raw short-duration markets are discarded immediately; the full raw corpus would not fit comfortably in memory).
4. Add `fetch_markets.py --full`: re-fetch everything from `START_DATE_MIN`, checkpointing each completed window to `data/fetch_cache/`, then merge with the existing meta CSV (existing rows win, so LLM categories and prior values are preserved; markets no longer returned by the API are kept).
5. After the full re-fetch, recompute the 80/20 temporal split on the full corpus with `scripts/recompute_split.py` (ADR-001 method unchanged; only the cutoff moves).
6. The meta CSV is the single source of truth for `split`. `fix_dataset.py` re-stamps every snapshot row's `split` from meta when writing the parquet, because `build_snapshots.py` only labels rows when they are first built and existing rows would otherwise keep the old assignment.

## Rationale

- Keyset pagination is the API's documented path for deep pagination and removes both caps at once.
- Failing loudly is the only way to guarantee the incremental watermark never advances past a gap.
- Keeping existing rows on merge avoids re-paying for categorisation and avoids silently changing labels on markets already in the training set.

## Alternatives considered

- **Keep offset pagination with `limit=100`** — fixes (1) but not (2); days with >3,000 markets still lose data.
- **Server-side duration filter via `end_date_min`** — would cut the volume of short markets fetched, but duration is computed from `createdAt` while windows filter on `startDate`, so no bound is provably safe. Rejected to avoid excluding valid markets.
- **Fetch only June 2026 onward correctly, backfill later** — leaves a known ~16% historical gap in training data; rejected by the user.

## Assumptions

1. `/markets/keyset` honours `closed`, `volume_num_min`, `start_date_min`, `start_date_max` (verified 2026-09-25 on a sample window).
2. `next_cursor` is absent or null on the last page (verified: 2025-03-01 → 03-15 exhausted after 16 pages, no duplicate IDs).
3. The ADR-008 filters are unchanged; only the fetch mechanism changes.

## Consequences

- **Easier:** complete market coverage; no silent data loss; resumable full fetch.
- **Harder:** a full fetch pages through every ≥$1k market since 2023, including hundreds of thousands of short markets that are filtered out client-side — expect a long run (tens of minutes to hours).
- **Split moves:** the recomputed cutoff changes which markets are train vs test. Metrics from before this ADR are not directly comparable with metrics after it; all models must be retrained and re-evaluated.
- **Risk:** the Gamma API may change again; the fail-loud behaviour will surface this rather than hide it.

## Related ADRs

- ADR-001: Temporal train/test split (method unchanged, cutoff recomputed)
- ADR-008: Market filters (unchanged)
