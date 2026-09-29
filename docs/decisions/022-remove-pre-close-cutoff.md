# ADR-022: Remove the 14-Day Pre-Close Snapshot Cutoff (Dataset v3)

**Date:** 2026-09-28  
**Status:** Accepted — applies from dataset v3; v2 keeps the 14-day cutoff  
**Deciders:** Hector Thompson Baroni  
**Supersedes:** ADR-002 (the 14-day cutoff only; 12h interval and 14-day burn-in stay)  
**Code location:** `data_collection_pipeline/build_snapshots.py` (`CUTOFF_BEFORE_CLOSE`, `--full`), `fix_dataset.py --replace`, `run_pipeline.py --rebuild-snapshots`

---

## Context

ADR-002 stopped snapshots 14 days before each market's scheduled `endDate`. The reason was that markets converge to 0 or 1 near the end, and those near-certain rows inflate AUC. ADR-002 itself noted that a price-based rule was meant to replace it. That rule now exists: ADR-014's leakage filter drops every snapshot priced ≥ 0.95 or ≤ 0.05, whenever it occurs.

With both rules in place, the fixed cutoff mainly throws away useful rows:

1. **Genuinely uncertain late snapshots are lost.** Example: a market at 0.55 five days before resolution. That is where a model could add the most value, and it's the most practically useful time to predict.
2. **It mismatches live use (ADR-017).** Users will look up markets that close soon, so the models would predict outside anything in their training data (`days_before_close` < 14).
3. **Test markets are lost.** Under ADR-021, 1,678 test markets contribute no rows in v2, partly because their snapshots stop two weeks before a July end date.

## Decision

- `CUTOFF_BEFORE_CLOSE = 0`: snapshots run up to the scheduled `endDate`.
- Snapshots also stop at the market's `closedTime` when known. `fix_leakage.py` Pass 1 drops post-close rows anyway, so this only avoids building rows that would be deleted (11,784 markets closed more than 14 days before `endDate`).
- Near-certain rows are removed only by the ADR-014 price rule (≥ 0.95 / ≤ 0.05).
- v3 is a **full rebuild** of every market (`run_pipeline.py --rebuild-snapshots`), not a merge into v2. Mixing rows built under both rules would make the dataset inconsistent.
- The build now retries failed price-history requests. A market that keeps failing is reported, not silently counted as "no history", and the build exits non-zero so it is retried rather than merged incomplete. A settings file next to the CSV stops a build from resuming a CSV made with different settings.

## Rationale

The price rule targets the actual problem (near-certain rows) directly. The fixed window was a rough proxy for it. Removing the proxy keeps the uncertain late rows and drops only the certain ones.

## Alternatives considered

- **Keep 14 days** — rejected for the reasons above.
- **A shorter fixed cutoff (e.g. 3 days)** — still an arbitrary proxy for the price rule, and still leaves a gap relative to live use.
- **Merge the extra rows into v2** — mixes two snapshot rules in one dataset; rejected.

## Assumptions

1. The ≥ 0.95 / ≤ 0.05 rule removes the rows where the outcome is effectively known. Rows at, say, 0.90 in the last hours remain and are legitimately hard to beat. The market baseline sees the same rows (ADR-015).
2. `endDate` is known in advance, so `days_before_close` stays a legitimate feature (ADR-014).

## Consequences

- v3 will have more rows per market near resolution. Class balance and the market baseline shift again, so recompute both; don't compare raw metrics with v2 without noting it.
- Comparing v2 and v3 models (same test cutoff rule, ADR-021) shows whether the late rows help. Report metrics broken down by `days_before_close` bucket as well.
- A full rebuild re-fetches price history for ~45k markets. `build_snapshots.py` now runs 4 worker processes (`--workers`). Measured on 200 markets: 0.91 s per market sequentially vs 0.375 s with 4 workers, with output identical to the sequential code. That makes a full rebuild about 5 hours instead of about 12. No CLOB request was refused at up to 16 concurrent requests.

## Related ADRs

- ADR-002: Snapshot window design (cutoff superseded here)
- ADR-014: Leakage filter (price rule now the only near-resolution filter)
- ADR-017: On-demand prediction (needs late-market coverage)
- ADR-021: Leak-free evaluation (split rule unchanged)
