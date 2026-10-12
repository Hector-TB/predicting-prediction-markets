# ADR-026: Daily Random Sample + Resolution Job for the Live Track Record

**Date:** 2026-10-12  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Code location:** `scripts/sample_markets.py`, `scripts/resolve_lookups.py`, `api/track_record.py`, `.github/workflows/live-track-record.yml`

---

## Context

ADR-017 stores every `/predict` lookup so that, once markets resolve, the site can show a live track record. Two things were missing:

1. **Nothing records outcomes.** Looked-up markets stay `resolution_status = 'open'` forever, so no lookup can be scored.
2. **Lookups are not a random sample.** People look up famous, contested markets (elections near 50/50), probably the hardest ones to beat the market on. A live result on those can't be compared with the backtest: a loss could mean "worse live" or just "harder markets".

## Decision

### 1. Daily random sample (`scripts/sample_markets.py`)

Each day, **20 markets** drawn uniformly at random from the eligible open markets are predicted with the production release and stored like a lookup, with `snapshots.source = 'sample'` (user lookups keep `'search'`).

**Eligible = a market the models were trained on (ADR-008, ADR-002, ADR-014), as far as it can be known while open:**

- from Gamma's open markets: binary, YES token, `startDate` ≥ 2023-01-01, volume to date ≥ $1,000, `endDate` in the future, scheduled duration (`endDate − createdAt`) ≥ 30 days, created ≥ 14 days ago (burn-in), Gamma YES price strictly between 0.05 and 0.95;
- not sampled before (each market enters the sample at most once);
- after computing its features, none of these ADR-025 warnings: `few_prices`, `young_market`, `settled_price`, `outside_filters`, `no_end_date`, `past_end_date`. A drawn market that fails is skipped (not stored) and the next one in the random order is tried.

`near_close` and `long_market` markets stay in: the backtest test set contains them too. `no_category` is allowed (rare; recorded as a warning).

The draw order is seeded by the UTC date, so a day's sample can be reproduced from the same Gamma listing.

### 2. Resolution job (`scripts/resolve_lookups.py`)

Daily, before the sample: every market with `resolution_status = 'open'` (sampled or looked up) is fetched from Gamma by condition ID in batches of 100. Closed ones get the training outcome rule (ADR-004): final YES price ≥ 0.95 → `outcome = 1`, ≤ 0.05 → `0`, `resolution_status = 'resolved'`; otherwise `ambiguous` with `outcome` NULL. `closed_time` (Gamma's `closedTime`) and `resolved_at` (now) are set. Markets Gamma doesn't return stay `open` and are counted in the log.

### 3. Scoring (`GET /track-record`)

- **One prediction per market:** its first lookup within each source, scored with the release that made it.
- **Sample and search are reported separately;** the sample is the headline, the search results carry the caveat that people chose those markets.
- Metrics: AUC, log-loss, Brier per model and for the market, with model − market differences and 95% bootstrap CIs (one row per market, so resampling rows resamples markets; ADR-015).
- **Market baseline = Gamma's YES price at the lookup** (`snapshots.market_price`), the price a trader saw. The backtest baseline is `price_at_snapshot`, the CLOB price at 12-hour fidelity, which can be up to 12 h stale; using it live would handicap the market.
- **Backtest comparison = the release's `per_market` metrics** (each market counts once), the closest analogue of one prediction per market.
- **Length bias:** on any day, long markets are over-represented among open markets (a year-long market is in the pool ~12× as many days as a 30-day one), and the backtest found no edge in markets over a year. So the sample is also reported **weighted by 1 / eligible days** (scheduled duration − 14-day burn-in), which undoes the over-representation, and by duration bucket (≤ 90 d, 90–365 d, > 365 d).
- Under **100 resolved markets** (or with only one outcome class) the endpoint returns counts only, no metrics.

### 4. Running it

One GitHub Actions workflow, daily: resolve, then sample. Secrets: `DATABASE_URL` (session pooler), AWS keys (release download), `ANTHROPIC_API_KEY` (category). Public repo: Actions minutes are free; logs are public (secrets masked; market IDs visible, which is fine).

## Rationale

The random sample turns the live track record into an out-of-time test of the backtest claim (r2 XGBoost per-market ΔAUC +0.012 [+0.009, +0.015] on v4): predictions made before anyone knew the outcome, on markets nobody chose. Restricting it to training-like markets means the live number measures what the backtest measured; markets the models were never meant for are already covered by ADR-025's warnings on user lookups.

**Sample size.** Measured on r2's XGBoost test predictions (v4, 5,457 markets) by drawing n markets with one random snapshot each, as the live sample does (300 draws per n): ΔAUC vs the market averages +0.012, with a 95% half-width of ±0.034 at 100 markets, ±0.019 at 300, ±0.010 at 1,000 and ±0.0055 at 3,000. So about 1,000 resolved markets are needed before an edge of this size can show clear of zero. 20/day ≈ 7,300 predictions a year, so a firm answer takes about a year (resolution lags behind market length). Storage ≈ 15–25 MB a year of the free tier's 500 MB.

## Alternatives considered

- **Every open binary market** as the population — broader, but mixes in cases where we already know the models have no edge or never saw such inputs; rejected for the headline (user lookups already cover it).
- **Length-stratified sampling** (draw with probability ∝ 1 / eligible days) — removes the length bias at the source, but the sample then isn't uniform and is harder to explain; reweighting at scoring time gives the same estimate and keeps the raw numbers.
- **Larger samples (100+/day)** — faster answers, but ~5× the Gamma/CLOB/Haiku calls and storage; 20 is enough for a firm answer within about a year. Revisit if the CI is still wide.
- **Score against `price_at_snapshot`** like the backtest — rejected, see the baseline point above.

## Assumptions

1. Gamma's `outcomePrices` of a closed market are final, as in the training fetch (ADR-004).
2. Volume to date ≥ $1k is a fair live stand-in for the training filter on final lifetime volume (ADR-008, ADR-018).
3. The 14-day burn-in and 30-day duration use `createdAt`, like `build_snapshots.py`.

## Consequences

- The live track record has an unbiased headline once ~100 sampled markets have resolved (months), firm at ~1,000.
- A release change mid-way splits the live record by release; each market is scored with the release that first predicted it.
- The daily DB activity also keeps the free Supabase project from pausing.
- Risk: if Gamma changes its open-market listing, the sample silently shrinks — the job logs the eligible count each day.

## Related ADRs

- ADR-004: Outcome threshold
- ADR-008: Market filters
- ADR-015: Evaluation methodology
- ADR-017: On-demand prediction + live track record
- ADR-025: Always predict, with warnings
