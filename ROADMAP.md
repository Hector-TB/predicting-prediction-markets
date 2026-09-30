# Roadmap

Ordered by priority. The reasoning behind each decision is in `docs/decisions/` (ADRs). This file is the plan; ADRs only record decisions.

**Rule:** any idea, follow-up or "later" item that comes up goes in here straight away: under the step it belongs to, or under *Parked* if it has no home yet. Update the file in the same change as the work.

---

## Done — the foundation

### 1. ~~Database~~ ✓
Schema in `db/migrations/001_initial_schema.sql` (5 tables). Holds v3 since 2026-09-29: 45,130 markets (Gamma dates), 1,755 trends rows, model runs for v1 and v3, loaded with `db/load_parquet.py`.

### 2. ~~Harden the data pipeline~~ ✓
- 2026-09-25–27: keyset pagination + full re-fetch (ADR-013); working closedTime lookup + per-row settled-price filter (ADR-014); LLM categories keyed by id; pipeline streams in batches to fit 8 GB of RAM
- 2026-09-28: six pipeline bugs fixed; **v2 published**; leak-free evaluation (ADR-021): split by resolution time, `log_volume` dropped, thresholds and CV off the test set
- 2026-09-28: **v3 published**: no 14-day pre-close cutoff (ADR-022); metadata matched to Gamma (ADR-023, checks block build and publish); coverage check 0 missed; full quality check passed
- 2026-09-29: incremental fetch by scheduled end date (ADR-020, PR #2). A dry run found 737 new markets, including all 692 the coverage check had flagged

### 3. ~~Retrain on v3 and re-score the paper~~ ✓ 2026-09-29 (PR #1)
- One training protocol for all models (ADR-024): time-ordered holdout, one weight per market, calibration bounded to [0.001, 0.999]
- LR, XGBoost and RF trained on v3; predictions on S3 (`predictions/v3/`, `scripts/sync_predictions.py`)
- Re-score (`docs/paper/rescore_v3_data.md`, README updated):
  - XGBoost/RF beat the market: +0.014 AUC, and +0.011 AUC / −0.013 log-loss with each market counted once, all CIs clear of zero; LR is level with the market;
  - the gain is ≥ 30 days before close, also within markets of the same length; none in markets over a year;
  - trading survives a 2¢/token cost (fills unverified)
- Supabase refreshed with v3; the DB loader's silent date blanking fixed

---

## Now

### 4. One-command refresh with model releases (ADR-019; design in its 2026-09-29 amendment)
`scripts/refresh.py`, stages:
0. preflight;
1. end-date fetch;
2. move T (80/20 by resolution time, every refresh);
3. build (new markets only);
4. data gates: meta checks, coverage over recent end dates (full recount automatically if the last is > 30 days old), snapshot check vs the parent version, smoke test;
5. auto-publish vN+1;
6. train with production's settings (`--tune` to re-search);
7. candidate release `rN` on S3;
8. change report vs production with promotion checks.

`scripts/release.py promote | reject | rollback` needs explicit confirmation.

Build order:
- [x] `release.py init`: package the current v3 models as `r1` (production) ✓ 2026-09-29. `s3://…/models/r1/`, record in `models/releases/`; each model's saved file reproduced its published v3 predictions
- [x] `train.py --params` / `--out-dir` and `predict(artifact, df)` for LR, XGBoost, RF ✓ 2026-09-29. Retraining v3 with r1's settings reproduces r1 exactly (LR/XGBoost identical, RF within CSV rounding); LR and XGBoost take ~1 min each, RF ~15 min
- [x] `check_snapshots.py --parent vN` (market-by-market vs the parent: identical rows apart from `split`; v3 vs itself: 39,588 markets identical) and `coverage_check.py --end-from` (recent window, ~40 min; 0 missed, 753 new since the v3 fetch) ✓ 2026-09-29. A passing full recount is recorded as `last_full_coverage` in `data/fetch_state.json`
- [ ] `refresh.py` stages 0–5 (data), resumable. Stage 3 first retires the previous build's raw CSV: after v3 it still holds the whole 1.7 GB `--full` build, which would block an incremental build (see the ADR-019 amendment)
- [ ] stages 6–8 (train, release, report) + promotion checks; migration 005 (`model_runs.release`)
- [ ] The DB loader only upserts: prune markets dropped from a new dataset version (the 13 ADR-023 drops were deleted by hand)
- [ ] First real run: v4 (the 737+ new markets) → candidate `r2`

---

## Next — search-and-predict with a live track record (ADR-017, `docs/design/search-and-predict.md`)

### 5. Prerequisites
- ~~Decide and apply ADR-018~~ ✓ `log_volume` dropped (ADR-021)
- Refactor the per-snapshot feature code in `build_snapshots.py` into a shared function used by training and the API; test it reproduces training snapshots exactly
- DB migration: full feature vector + warnings on `snapshots`; resolution status on `markets`

### 6. FastAPI backend
- `GET /resolve?q=` — URL / slug / event / condition ID / name → market(s)
- `GET /predict?market_id=` — features as of now → the production release's models (LR, XGBoost, RF) + gap vs price, with warnings for out-of-distribution markets (non-binary rejected); stores every lookup with its release
- `GET /track-record` — backtest metrics + live metrics over resolved lookups, per release
- Daily resolution job: record outcomes of looked-up markets (ADR-004 rule)
- The models only beat the market ≥ 30 days before close (re-score). Label predictions closer to close as "no expected edge", or de-emphasise them

---

## Later — frontend

### 7. React frontend (public portfolio site)
Search → market detail page (the production models + gap, prediction history, outcome), recent predictions feed (live track record), model pages (importance, calibration, backtest vs live), track-record comparison, research page. Blocked on: API.

---

## Parked — to do, not yet scheduled

- **Walk-forward backtest.** Train at each quarterly cutoff from 2025 Q1, predict the next quarter, and pool the results. It shows:
  - whether the edge over the market is stable across periods;
  - when the models fail;
  - how fast they go stale (refresh cadence);
  - the normal quarter-to-quarter swing, for reading the live track record.

  Reuses `train.py --params` and `predict()`, so it comes after step 4. Run it once, then whenever settings are re-tuned or features change, never inside routine refreshes.

  Two rules (Dhar & Stein 1998; Sobehart, Keenan & Stein 2000):
  - never develop the model against walk-forward results, or they stop being out-of-time;
  - only point-in-time inputs, so Google Trends stays out until its rework.

  Gets its own ADR when built.
- **Rework Google Trends.** Category-level keywords, one 0–100 scale over the whole period (the scale depends on later peaks, so it isn't point-in-time), weekly granularity (ADR-009 amendment). The Trends model variants stay untrained until then; RQ2 of the paper waits for it.
- **Volume to date.** Cumulative traded volume up to each snapshot from `data-api.polymarket.com/trades` (ADR-018 option 2). That endpoint caps at ~10.5k trades per market; large markets need another source (e.g. the orderbook subgraph). New ADR + dataset version.
- **Realistic trading fills.** The trading simulation assumes fills at the recorded price. Large model–market gaps are probably more common in thin markets, where that price may be a stale last trade. Needs order-book / spread history.
- **Incremental merge and leakage steps.** They rewrite the whole parquet each run (~20–40 min). Only if it becomes a problem.
- **Clean-up** (no rush):
  - `data/polymarket_ml_dataset.v2.csv`, `data/*.bak.csv`, `data/*_part[12]*.parquet`;
  - S3 `staging/2026-09-27/` and `staging/2026-09-28-v3/`.

---

## Ongoing

- Keep ADRs up to date for any new architectural decisions
- Add new ideas and follow-ups to this file as they come up (see the rule at the top)
- Run `python scripts/smoke_test.py` after any pipeline changes
