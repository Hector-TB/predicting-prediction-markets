# ADR-008: Market Filters ($1k volume, 30d duration, 2023 start, binary only)

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/fetch_markets.py:37–41`

---

## Context

The Polymarket Gamma API returns all closed markets, including very thin markets (near-zero volume), very short markets (same-day bets), pre-CLOB markets (before the CLOB API existed), and non-binary markets (multiple outcomes). We need principled filters.

## Decision

Server-side filters (applied in API request):
- `volume_num_min = 1,000` — minimum lifetime trading volume of $1,000 USD
- `start_date_min = "2023-01-01"` — only CLOB era markets

Client-side filters (applied after fetch):
- `MIN_DURATION_DAYS = 30` — market must have lasted at least 30 days
- Binary only — exactly 2 outcomes (YES/NO)
- Unambiguous outcome — final price ≥ 0.95 or ≤ 0.05 (see ADR-004)
- Valid `closedTime` and a YES CLOB token ID must exist

## Rationale

**$1k volume floor:** Eliminates markets with no real trading activity. Markets below $1k have thin order books; their prices may not be informative. Also reduces noise in the rolling price features.

**2023-01-01 start:** The CLOB (Central Limit Order Book) API — which provides 12-hour price history — was launched in 2023. Pre-CLOB markets cannot be enriched with price history.

**30-day minimum duration:** Very short markets (< 30 days) don't have enough history for the 14-day burn-in and 14-day cutoff simultaneously (they'd be dropped in `build_snapshots.py` anyway). Filtering here avoids pointless API calls.

**Binary only:** Polymarket has a few multi-outcome markets (e.g., "which team wins the World Series" with 30 outcomes). These require a different modelling approach and are out of scope.

## Alternatives considered

- **$500 volume floor** — would include ~10% more markets but with very thin liquidity; rejected
- **$10k volume floor** — would remove high-quality low-volume markets (e.g., early niche markets); rejected
- **No duration filter** — would let build_snapshots.py handle it implicitly; acceptable but less clear
- **Include pre-CLOB markets** — would require a different price data source; out of scope

## Assumptions

1. Markets with <$1k volume have prices that are not informative about the real probability
2. 2023-01-01 is a reliable proxy for CLOB availability
3. 30 days is the minimum meaningful duration for rolling price features

## Consequences

- 24,092 markets pass all filters out of a larger raw fetch
- 3,144 (13%) are additionally dropped during `build_snapshots.py` for sparse price history
- Distribution is skewed toward sports (29.4%) and US politics (15.6%)

## Related ADRs

- ADR-004: Outcome threshold (part of the filter chain)
- ADR-001: Train/test split (applied to the filtered set)
