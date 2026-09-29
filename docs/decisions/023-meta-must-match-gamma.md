# ADR-023: Market Metadata Must Match Gamma — No Estimated Values

**Date:** 2026-09-28  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data_collection_pipeline/meta_checks.py`, `fetch_markets.py` (`merge_fresh`, `latest_fetch.csv`), `scripts/refresh_meta.py`, checks wired into `run_pipeline.py`, `build_snapshots.py` and `data/sync.py publish`

---

## Context

A quality check on the first v3 snapshots (ADR-022) found that 14,886 of 45,143 meta rows (33%) had **estimated** dates, not Polymarket's:

1. **2026-09-19:** a re-fetch missed about 15k markets (the offset-cap bug fixed in ADR-013). `scripts/rebuild_meta_from_parquet.py` recreated their meta rows from the old snapshot parquet, and estimated the fields snapshots don't store:
   - `start_date`, worked back from the first snapshot and `pct_lifetime_elapsed`
   - `end_date`, as start + a whole number of days
   - an empty `clob_token_id`
   - `yes_final_price` set to 0.05 / 0.95
2. **2026-09-25:** the fixed full re-fetch returned these markets with their real values. But `fetch_markets.py --full` merged with **existing rows winning**, a rule meant to protect LLM categories. So the estimates were kept and the real values discarded.
3. **2026-09-28:** the placeholder token and price were patched, but the dates were left alone because v2's snapshots were built from them.

Effects:
- 9,742 start dates off by more than 1 hour.
- 14,259 end dates off by more than 1 hour, 61 by more than a day (with wrong `duration_days`).
- 13 markets whose real lifetime is 16–27 days, so they should have failed the 30-day filter (ADR-008), but estimation made them look 32–381 days long.

v2's *snapshots* were unaffected: they came from the April build, which used the real dates. v2's published meta CSV does contain the estimated dates, and the 13 too-short markets are in v2.

Nothing caught this because an estimated row is internally consistent: its duration matches its invented end date.

## Decision

1. **Every Gamma field in meta comes from Gamma, unchanged.** Only `category` (LLM, ADR-008) and `split` (ADR-021) are ours. No field may be estimated, back-filled or given a placeholder. A market whose values can't be fetched is dropped, not approximated.
2. **Merges take fresh values.** `fetch_markets.merge_fresh()`, used by both full and incremental fetches, replaces every Gamma field of a re-fetched market and keeps only `category` and `split`.
3. **Keep the evidence.** Each fetch saves its Gamma records to `data/fetch_cache/latest_fetch.csv`; `scripts/refresh_meta.py` saves direct lookups to `lookups.csv`. When files overlap, the newest wins.
4. **Checks that block the pipeline** (`meta_checks.py`):
   - Internal: no empty required fields; no 0.05 / 0.95 placeholder prices; outcome consistent with the final price; `duration_days` = (end − start) in days; valid split; unique ids.
   - Against Gamma: every Gamma field equals the saved Gamma record, and every market has one. This is the only check that catches plausible invented values.
   - Enforced in `run_pipeline.py` (after the fetch), `build_snapshots.py` (before building), `fetch_markets.py` (before saving a merge) and `sync.py publish` (before upload).
5. **Retire the scripts that produced or patched estimates:** `rebuild_meta_from_parquet.py` and `patch_backfilled_meta.py`. `scripts/refresh_meta.py` is the repair tool: it re-fetches from Gamma and never estimates.

## Rationale

Metadata feeds snapshot windows, time features, filters and the split. An estimated value silently changes all of them. Refusing to run is cheaper than finding out after a day-long build, which is how this was found.

## Alternatives considered

- **Patch only the dates** — fixes this instance, not the cause. The merge rule would reintroduce stale values on the next fetch.
- **Keep unverifiable markets with a flag** — the models would still train on invented values. Rejected: drop them.

## Consequences

- Meta refreshed on 2026-09-28: 45,130 markets. The 13 too-short markets were dropped; 14,795 start dates, 14,887 end dates, 61 durations, 227 volumes and 2 questions were updated. No outcome changed. The split was recomputed: T is unchanged (2026-07-01 07:11 UTC) and 3 markets changed side.
- The partial v3 build was discarded, and v3 is rebuilt from the refreshed meta.
- v2 stays as published (immutable, ADR-016). Its meta CSV has estimated dates for 14,886 markets and includes the 13 too-short markets. Its training snapshots used real dates. Note this when comparing v2 and v3.
- A fetch cache is now required to build or publish.

## Related ADRs

- ADR-008: Market filters (30-day minimum the 13 markets failed)
- ADR-013: Full re-fetch (the merge rule is corrected here)
- ADR-016: Dataset versions (v2 unchanged)
- ADR-021: Split (recomputed after the refresh)
- ADR-022: v3 rebuild (where this was found)
