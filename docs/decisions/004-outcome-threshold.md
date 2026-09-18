# ADR-004: Outcome Threshold (0.95 / 0.05)

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/fetch_markets.py:40` — `OUTCOME_THRESHOLD = 0.95`

---

## Context

Polymarket markets settle by the `outcomePrices` field in the Gamma API, which reports the final price of the YES token. In theory this is exactly 1.0 (YES) or 0.0 (NO), but in practice:
- Some markets settle between 0.95 and 1.0 due to API timing or partial resolution
- Some markets are genuinely ambiguous (e.g., the outcome was disputed or N/A)

We need a clear rule for assigning ground-truth labels.

## Decision

- `yes_final_price >= 0.95` → `outcome = 1` (YES)
- `yes_final_price <= 0.05` → `outcome = 0` (NO)
- Anything in between → **drop** (labelled "ambiguous")

## Rationale

0.95/0.05 is a conservative choice. It catches markets that have settled but where the API reports 0.97 instead of 1.0 due to rounding or timing. The alternative — requiring exactly 1.0/0.0 — would drop many legitimately resolved markets.

**Ambiguous markets (between 0.05 and 0.95) are dropped** rather than assigned a label. Including them with a noisy label would degrade model quality. The README notes these are a small minority.

## Alternatives considered

- **Threshold 0.99** — too strict; drops many valid resolutions due to API imprecision
- **Threshold 0.90** — too loose; would label some genuinely ambiguous markets
- **Keep ambiguous markets as a third class** — prediction markets are binary; a third class adds complexity with no research value
- **Use `resolved` field directly** — the Gamma API's `resolved` boolean is less reliable than the price field for our purposes

## Assumptions

1. Any market where the YES token settles above 0.95 is effectively a YES resolution
2. Markets between 0.05 and 0.95 are genuinely ambiguous (disputed outcomes, early resolution, etc.)

## Consequences

- A small number of legitimately resolved markets may be dropped if their API settlement price is between 0.05 and 0.95
- Outcome distribution: ~19.7% YES, ~80.3% NO — significant class imbalance handled separately (ADR-007)

## Related ADRs

- ADR-007: Class imbalance handling
- ADR-008: Market filters
