# ADR-003: Rolling Feature Windows (7d and 14d)

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/build_snapshots.py:39` — `ROLLING_WINDOWS = [7, 14]`

---

## Context

We need rolling statistical features (mean, volatility, min, max, change, range, slope) to capture price dynamics. The window length determines what timescale of signal we capture.

## Decision

Two windows: **7 days** and **14 days**.

Features per window: `price_mean`, `price_volatility` (std), `price_min`, `price_max`, `price_change` (last − first), `price_range` (max − min), `price_trend` (OLS slope).

## Rationale

**Why two windows:** A single window cannot distinguish short-term momentum from medium-term trend. Having both allows models to detect *divergence* — e.g., a price rising over 7 days that is still below its 14-day mean signals a recovery, not a breakout.

**Why 7d:** Captures within-week dynamics — news events, weekend effects, weekly cycles in sports markets.

**Why 14d:** Two-week window smooths the noisier 7d signal and captures slower drift. Also aligns with our burn-in (ADR-002): every snapshot we keep has at least 14 days of history, so the 14d window is always fully populated.

**Feature correlations with outcome (from README):**

| Feature | r with outcome |
|---|---|
| price_at_snapshot | +0.643 |
| price_mean_7d | +0.637 |
| price_mean_14d | +0.627 |
| price_change_14d | +0.217 |
| price_trend_14d | +0.157 |

Rolling means are nearly as informative as the current price. Momentum features (change, trend) carry modest but independent signal.

## Alternatives considered

- **30-day window** — would miss short-term dynamics and reduce available history for early snapshots; rejected
- **3-day window** — too noisy; microstructure noise dominates; rejected
- **Single window (14d only)** — loses the divergence signal between short and medium term; rejected

## Assumptions

1. 7 and 14 days are the most informative timescales for Polymarket markets (supported empirically but not exhaustively searched)
2. OLS slope (`price_trend`) is a useful proxy for directional momentum even with only ~14 data points

## Consequences

- 14 features per window × 2 windows = 28 rolling features (plus base features = ~27 total columns)
- `price_mean_{7d,14d}` are highly correlated with `price_at_snapshot` (r > 0.96 in SVM analysis) — SVM explicitly drops them to reduce redundancy; tree models tolerate this

## Related ADRs

- ADR-002: Snapshot window design (burn-in ensures windows are always populated)
