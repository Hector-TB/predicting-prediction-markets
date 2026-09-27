# ADR-017: On-Demand Prediction via Search, with a Stored Live Track Record

**Date:** 2026-09-27  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Design doc:** `docs/design/search-and-predict.md`

---

## Context

The roadmap planned a batch live-scoring pipeline: fetch every open market, score it on a schedule, and show a browsable list. On 2026-09-27 there were about 6,700 open markets our models could score (binary, at least 14 days old, at least 30 days' duration, at least $1k volume, price between 0.05 and 0.95, and at least 14 days before scheduled end), out of about 12,100 open markets with at least $1k volume.

The site's purpose is a public portfolio piece. What matters is showing what the models say about a market someone cares about, and proving over time whether they were right.

## Decision

1. **No batch live scoring and no market list.** Users search for a market by URL, slug, event, condition ID or name. The API fetches that market, computes its features **as of now** with the training feature code, and returns every model's probability and its gap to the market price. One headline model is shown first, with the rest below.
2. **Eligibility:** non-binary (and already-closed) markets are **rejected**. Markets outside the training distribution (too young, too close to their end date, near-settled price, below the market filters, stale Trends) are **predicted with explicit warnings**.
3. **Every lookup is stored.** The complete feature vector, the warnings, the market price and every model's probability go into `snapshots` / `predictions`. A daily job records each market's outcome when it resolves.
4. **The live track record is scored with ADR-015's rules:** the first lookup per market, all models on the same markets, market-level bootstrap CIs, and results reported with and without warned predictions.

## Rationale

- One market at a time needs a small API instead of a scheduled job over thousands of markets, and every Polymarket call is one a user asked for.
- A stored, timestamped prediction that is later checked against the outcome is the most credible evidence the project can offer: it cannot be tuned after the fact, unlike a backtest.
- Warnings rather than refusals keep the tool useful, while the stored warning flags keep the track record honest.

## Alternatives considered

- **Batch scoring + market list** — more infrastructure for a list no one needs to browse; rejected.
- **Refuse all out-of-distribution markets** — cleaner statistically, but frustrating for users; warnings plus separate reporting achieves the same rigor.
- **Don't store lookups** — would throw away the only out-of-sample evidence the site can generate; rejected.

## Assumptions

1. The Gamma `slug`, `condition_ids`, `events` and `public-search` lookups keep working as verified on 2026-09-27.
2. Features computed at lookup time from the full CLOB history match what `build_snapshots.py` would compute for a snapshot at that time. This must be enforced by sharing the code and tested.
3. Looked-up markets are a biased sample (people search for interesting markets). The live track record says how the models did on markets users chose, not on Polymarket at large; the site must say so.

## Consequences

- Roadmap step 5 (batch live scoring) is replaced by: a shared feature function, `/resolve` + `/predict` endpoints, the storage migration, and the resolution job.
- **Blocked on ADR-018:** the volume features currently use each market's final lifetime volume, which doesn't exist for an open market.
- The live track record needs time: meaningful metrics only after ~100+ looked-up markets have resolved.

## Related ADRs

- ADR-002, ADR-008, ADR-014: define the training distribution the warnings refer to
- ADR-004: outcome threshold used by the resolution job
- ADR-010 / ADR-011: `snapshots` / `predictions` tables for live data
- ADR-015: evaluation rules applied to the live track record
- ADR-018: volume feature (prerequisite)
