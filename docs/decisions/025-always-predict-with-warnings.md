# ADR-025: Always Predict an Open Binary Market; Flag What Training Never Saw

**Date:** 2026-10-09  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni

---

## Context

ADR-017 decided that markets outside the training distribution are predicted with warnings, and listed the cases. Two cases were left open, because the time features (`days_before_close`, `pct_lifetime_elapsed`, `duration_days`) need a scheduled end date:

- **No `endDate`**: 11 of 500 open markets sampled on 2026-10-01 (e.g. the 2026 Senate-seat markets). Training drops these (`fetch_markets.py`, `bad_dates`).
- **`endDate` already passed, market still open** (usually waiting to be resolved): 39 of 500. Every training row was taken at or before its market's end date (`days_before_close ≥ 0`).

The user's rule (2026-10-09): every open binary market gets a prediction; anything unusual is flagged instead of refused.

To see what a prediction is worth without an end date, production r1 scored the v4 test set (227,280 rows, 5,457 markets, none seen in training) twice: with the real time features, and with all three set to the training medians (87.2 days to close, 0.4365 elapsed, 173-day duration):

| | AUC | log-loss |
|---|---|---|
| Market price | 0.7928 | 0.5254 |
| XGBoost: real / medians | 0.8061 / 0.8026 | 0.5213 / 0.5366 |
| Random forest: real / medians | 0.8053 / 0.8025 | 0.5212 / 0.5330 |
| Logistic regression: real / medians | 0.7969 / 0.8016 | 0.5294 / 0.5228 |

With the medians, the tree models still rank better than the price (ΔAUC +0.010, market-level CI [+0.003, +0.015], 200 resamples), but their probabilities are worse than the market's (higher log-loss). These are markets that do have end dates; real no-end-date markets lean long-running, where the v3 re-score found no edge.

## Decision

1. **Every open binary market gets a prediction from every production model.** Only non-binary and already-closed markets are rejected (ADR-017).
2. **Missing inputs are filled for the models, never for the market.** The value fed to the models is stored with the lookup's feature vector, and the warning names it. The market's own fields keep Gamma's values (an absent `endDate` stays absent; ADR-023).
   - **No `endDate`** → `no_end_date`: the three time features are set to the medians of the release's training rows, stored with the release (a manifest field from r2 on; for r1, a file next to its record in `models/releases/`).
   - **`endDate` passed** → `past_end_date`: `days_before_close` = 0 and `pct_lifetime_elapsed` = 1, the closest values training saw.
   - **End date not after the creation date** → treated as no end date (training drops these as `negative_duration`).
   - **Fewer than 20 price points** → `few_prices`: features from the points there are (`features_at` gets a minimum-count argument; training keeps 20). With no CLOB history at all, the price is Gamma's current price and the rolling features are filled as for an empty window (`finish_features`).
3. **Warnings** (codes stored with every lookup; ADR-017 scores the track record with and without them):

   | Code | Condition | Why |
   |---|---|---|
   | `no_end_date` | no scheduled end date | time features are training medians: ranking holds, probabilities are worse than the market's (table above) |
   | `past_end_date` | scheduled end has passed | training never saw a snapshot after the end date |
   | `few_prices` | < 20 price points | below training's minimum |
   | `young_market` | < 14 days since `startDate` | training's burn-in (ADR-002) |
   | `settled_price` | price ≥ 0.95 or ≤ 0.05 | removed from training as settled (ADR-014) |
   | `near_close` | < 30 days to scheduled end | no edge over the market measured this close (v3 re-score) |
   | `long_market` | scheduled duration > 365 days | no edge measured in markets over a year |
   | `outside_filters` | duration < 30 days or volume below the market filter | outside training's market filters (ADR-008) |
   | `no_category` | the category call failed | the models get `other`, as for an unknown category in training |

   The design doc's "< 14 days to scheduled end" warning is replaced by `near_close`: ADR-022 removed the 14-day cutoff, so those rows are in training.

## Rationale

- A refusal tells the user nothing; a prediction with a precise warning lets them judge it, and the stored warning keeps the track record honest.
- Filling only the model input, and saying so, keeps ADR-023's rule: no estimated value is ever stored as a fact about the market.
- Medians were measured, not assumed, to keep ranking power; the warning text reflects the measured cost.

## Alternatives considered

- **Refuse these markets** — rejected by the user: always give a prediction.
- **Feed the raw values** (negative `days_before_close` past the end; the imputer's 0 for a missing end) — 0 days to close for a market with no end date says "closing now", the opposite of the truth; negative values are outside anything the models were fitted on.
- **Models without time features as a fallback** — more principled for `no_end_date`, but another model per release for ~2% of lookups. Parked in ROADMAP; the live track record will show whether it's needed.
- **Show the market price as the "prediction"** — not a model prediction; the gap would always be zero.

## Assumptions

- Open markets resemble the sampled 500: about 2% have no end date, about 8% are past it.
- The training medians stay a sensible middle as releases change (recomputed per release).

## Consequences

- `/predict` never fails for an open binary market with a price; warnings carry the caveats.
- Track-record reports must split by warning, or warned lookups (especially `no_end_date`, `settled_price`) would blur the measured edge.
- Releases need the median fill values: `release.py candidate` stores them from r2 on.

## Related ADRs

- ADR-002: snapshot window (burn-in)
- ADR-008: market filters
- ADR-014: settled-price rule
- ADR-017: on-demand prediction with warnings
- ADR-022: no pre-close cutoff
- ADR-023: meta must match Gamma
