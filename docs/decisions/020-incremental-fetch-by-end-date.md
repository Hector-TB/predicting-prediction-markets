# ADR-020: Incremental Market Fetch by Scheduled End Date, with a Look-Back

**Date:** 2026-09-28  
**Status:** Accepted — to be implemented with the one-command refresh (ROADMAP 4b)  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data_collection_pipeline/fetch_markets.py` (`load_existing_meta`, `_fetch_window`, `fetch_filtered_markets`)

---

## Context

`fetch_markets.py` requests closed markets filtered by **start date**. In incremental mode it only requests markets that started after (newest start date in the meta CSV − 7 days). A market that started months ago and closed since the last run is never requested, so long-running markets are silently missed. The `--full` fetch's monthly cache (`data/fetch_cache/`) has the same flaw: a cached month is reused even though markets that started in it keep closing afterwards.

v2 is not affected in practice: its `--full` fetch (2026-09-25/27) re-requested every month from 2023, so it includes everything closed by then (a possible gap is limited to markets that closed during the fetch itself).

Checked against the Gamma `/markets/keyset` API on 2026-09-28:

- There is no working filter on actual close time: `closed_time_min/max` is accepted but ignored (it returned the same 2020–2021 markets as no filter).
- `end_date_min/max` (**scheduled** end date) works, but only 56% of closed markets with a scheduled end in a given week actually closed in that week. Markets often resolve early (a "by Dec 31" market resolving in June) or late.

## Decision

1. The incremental fetch requests **closed markets whose scheduled end date is on or after (last fetch date − 30 days), with no upper bound**, plus the existing volume filter. The other filters are unchanged (created ≥ 2023-01-01, binary, duration ≥ 30 days, clear outcome), and markets already in the meta CSV are dropped by ID.
   - **Closed on time:** scheduled end falls in the range.
   - **Closed early:** scheduled end is in the future, which the open upper bound includes.
   - **Closed late,** up to 30 days after the scheduled end: covered by the look-back.
2. The last fetch date is stored explicitly (not inferred from the newest start date).
3. **Fetch caches are never reused across runs.** A monthly cache is only for resuming an interrupted `--full` run and is deleted when that run completes.
4. **Coverage check before publishing** (ADR-016): an independent recount partitioned by scheduled end date, using the same filters, must find no passing markets missing from the meta CSV. Otherwise `publish` refuses.

## Rationale

We only use markets once they have closed, so the fetch has to ask about markets by when they close, not when they started. Scheduled end date is the closest filter the API supports; the open upper bound and the look-back cover early and late closes.

## Alternatives considered

- **Re-run `--full` every time** — complete but slow (two days for v2); rejected for routine refreshes.
- **Filter by start date with a long look-back** — any finite look-back still misses markets longer than it; rejected.
- **Track open markets and poll them until they close** — exact, but needs a watchlist of all open markets; unnecessary given the approach above.

## Assumptions

1. Almost no market closes more than 30 days after its scheduled end. The pre-publish coverage check would catch any that do; lengthen the look-back if it ever finds some.
2. The Gamma `end_date_min` filter keeps its 2026-09-28 behaviour.

## Consequences

- Each incremental fetch returns some already-known markets (closed early with future scheduled ends); they are dropped by ID, at a modest cost.
- Implemented as part of the one-command refresh (ADR-019 / ROADMAP 4b).

## Related ADRs

- ADR-013: Keyset pagination + full re-fetch
- ADR-016: Dataset versioning (publish-time checks)
- ADR-019: Model releases / one-command refresh
