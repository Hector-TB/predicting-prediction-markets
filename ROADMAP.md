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
- [ ] `refresh.py` stages 0–5 (data), resumable — code written 2026-09-29 (`scripts/refresh.py`); preflight tested; full run = the first real refresh (below). Stage 3 first retires the previous build's raw CSV: after v3 it still holds the whole 1.7 GB `--full` build, which would block an incremental build (see the ADR-019 amendment)
- [ ] stages 6–8 (train, release, report) + promotion checks; migration 005 (`model_runs.release`) — code written 2026-09-30 (`release.py candidate | report | promote | reject | rollback | sync-db`, `scripts/release_report.py`; choices in ADR-019's *Implementation choices*). Dry run on v3 passed 2026-09-30 (r1 packaged as a pretend r2: artifacts reproduce, zero differences, verdict PASS; the report takes ~6 min at 200 resamples). Migration 005 applied and `release.py sync-db r1` run ✓ 2026-10-09
- [ ] Incremental build retries every parent market without rows (5,542 on v3 → 6,311 attempts for 769 new markets, ~30 min). Harmless (none gained rows on the first refresh; the parent gate would catch one), but it should build only markets not in the parent's meta
- [ ] The DB loader only upserts: prune markets dropped from a new dataset version (the 13 ADR-023 drops were deleted by hand)
- [x] First real run: v4 (769 new markets) → candidate `r2`. 2026-10-02: stages 0–5 done, **v4 published** (cutoff T = 2026-07-02 15:09 UTC; all four gates passed); stage 6 trained LR and XGBoost, then RF died mid-fit (run `logs/refresh_20260930_014318`). **Paused 2026-10-09** to build the API/site first (r1 stays production; the API serves whichever release is production). Finished 2026-10-12 with `--resume` (RF 19 min, release 10 min, report 9 min): **r2 candidate, verdict PASS** (ΔAUC vs market LR +0.0046, XGBoost +0.0143, RF +0.0128). **r2 promoted 2026-10-12** (vs r1: ΔAUC LR +0.0006 [+0.0002, +0.0010], XGBoost +0.0011 [+0.0004, +0.0017], RF +0.0003 [−0.0001, +0.0007]; `model_runs` rows written by `promote`); v4 markets loaded into Supabase (+769)

---

### 4b. Library pins and the version upgrade (ADR-019 amendment, 2026-10-12)
- [x] r1/r2 couldn't load under the new machine's scikit-learn 1.9.1 (`pyproject.toml` only had minimums). Exact pins for the release libraries, versions recorded per release and checked on load (API, `release.py`), `promote` reproduces published predictions first, `tests/test_environment.py`
- [x] **r3 promoted 2026-10-12**: v4 retrained with r2's settings under the pinned versions (LR 26 s, XGBoost 26 s, RF 6 min; peak 3.7 GB, needed the WSL raise to 6 GB). vs r2: LR identical, XGBoost AUC 0.8067 vs 0.8072, RF 0.8057 vs 0.8056, all checks pass; `promote` reproduced 2,000 published predictions per model. r1/r2 now load only under their recorded versions (`models/releases/r{1,2}_environment.json`)

---

## Next — search-and-predict with a live track record (ADR-017, `docs/design/search-and-predict.md`)

### 5. Prerequisites
- ~~Decide and apply ADR-018~~ ✓ `log_volume` dropped (ADR-021)
- [x] Shared feature code ✓ 2026-10-09: `data_collection_pipeline/features.py` (CLOB fetch, `raw_features`, `finish_features`, `features_at` for the API); `build_snapshots.py` and `fix_dataset.py` use it. `scripts/check_feature_parity.py` re-fetches sampled markets and recomputes every stored snapshot: 250 v4 markets / 22,808 rows, all 19 model features identical (a 12 h timestamp shift fails it). Re-run it after any feature change or library upgrade
- [ ] `check_feature_parity.py` compares exactly, but on the WSL2 machine ~0.02% of rows differ by 1e-6 in `price_trend_7d/14d` (old and new library versions alike: floating-point noise tips the 6-decimal rounding of a least-squares slope; 2026-10-12, ADR-019 amendment). Allow one rounding step for rounded features so a real change still fails it
- [x] DB migration written: `db/migrations/006_live_lookups.sql` (features + warnings on `snapshots`, resolution status on `markets`); applying it is under step 6
- [x] `/predict` endDate handling ✓ 2026-10-09 (ADR-025): always predict; no `endDate` → time features from the release's training medians (`no_end_date`), passed → 0 days left (`past_end_date`); full warning list in ADR-025 and the design doc. Measured: r1 on the v4 test set with median time features still out-ranks the price (ΔAUC +0.010 [+0.003, +0.015]) but its log-loss is worse than the market's
- Still open from that item: check whether Polymarket edits `endDate` while a market is open (meta stores the final value, so training time features could see an end date not yet known; cf. ADR-018). Lookups store the `endDate` seen at the time, so comparing with the final value answers it
- [x] Serving support ✓ 2026-10-09: `features_at(..., min_obs=)` and no-end-date path (parity re-checked); median fill values in every new release manifest (`time_feature_fill`), r1's in `models/releases/r1_time_fill.json` (from v3's 2,038,895 train rows, file sha-checked)

### 6. FastAPI backend
- [x] App skeleton ✓ 2026-10-09: `api/` (`uvicorn api.main:app`): `/health`, `/resolve` (URL / event / slug / condition ID / search), `/predict` (production release, ADR-025 fills + warnings, category via one Haiku call, cached per process). Release downloaded once into `.cache/releases/` (sha-checked); `$RELEASE` pins one. Offline tests in `tests/test_api.py` (`python3 -m pytest tests/`). Tried live: an ordinary market (~3 s), no end date, past end date, closed, unknown ID
- [x] Store every lookup — code ✓ 2026-10-09 (`api/store.py`, migration `006_live_lookups.sql`): market upserted as `open` (dataset markets untouched), snapshot with full features / warnings / Gamma price / `endDate` as seen / release, one prediction per model linked to the release's `model_runs`. If the DB is down the prediction is still returned with `"stored": false` and the miss is logged. Tested end to end on a throwaway Postgres (PGlite) with migrations 001–006
- [x] Applied to Supabase ✓ 2026-10-09 (after an unpause; tables intact): 005 + 006 in one transaction, RLS on for all five tables, r1 registered in `model_runs`. First real lookup stored (snapshot 1, 3 predictions)
- [ ] Limit lookup storage before going public (rate limit per IP, or reuse a lookup of the same market within a few hours): the free tier has 500 MB
- [ ] Deploy: where the API runs (needs S3 + Anthropic keys, ~220 MB peak RAM with r1 loaded, measured 2026-10-09)
- `GET /resolve?q=` — URL / slug / event / condition ID / name → market(s)
- `/predict` always predicts and attaches ADR-025's warning codes (stored with each lookup)
- `GET /predict?market_id=` — features as of now → the production release's models (LR, XGBoost, RF) + gap vs price, with warnings for out-of-distribution markets (non-binary rejected); stores every lookup with its release
- `GET /track-record` — backtest metrics + live metrics over resolved lookups, per release
- Daily resolution job: record outcomes of looked-up markets (ADR-004 rule)

---

## Later — frontend

### 7. React frontend (public portfolio site)
Search → market detail page (the production models + gap, prediction history, outcome), recent predictions feed (live track record), model pages (importance, calibration, backtest vs live), track-record comparison, research page.
- [x] First pages ✓ 2026-10-09 (`web/`, Vite + React + TS; run instructions in `web/README.md`): search (`/?q=`, URL / slug / text via `/resolve`) and market page (`/market/:id`: market price, headline model, gap, ADR-025 caveats, price chart with the model's estimate, all models, model inputs, stored-lookup note, "Predict again"). Checked in light and dark, desktop and phone width
- [ ] Market page history: earlier lookups of the market and, once resolved, the outcome — needs `GET /markets/{id}` (design doc)
- [ ] Recent-lookups feed + track-record page (`GET /lookups`, `GET /track-record`); model pages (`GET /models/{name}`)
- [ ] **"Does a gap like this pay off?"** on the market page (asked 2026-10-09 about a +7.7 pt gap): from the backtest test set, bin snapshots by the model–market gap (and days to close), show each bin's realised YES rate vs price and the return of buying the cheaper side at the recorded price (ADR-015: NO costs 1 − p), with market-level CIs; later the same from the live track record. Needs the bins computed per release (manifest or a small table) and an API field
- [ ] A page view stores a lookup: decide reuse vs new lookup on the server (with the storage-limit item in step 6) before going public

---

## Parked — to do, not yet scheduled

- **Time-free fallback models for `no_end_date` lookups** (ADR-025 alternative): models trained without the three time features. Only if the live track record shows the median fill hurts; ~2% of lookups.

- **Run refreshes on a cloud machine.** A refresh takes ~2 h and dies if the laptop sleeps or VS Code quits. An EC2 instance in the bucket's region (or another scheduled runner) could run `refresh.py` unattended, e.g. weekly. Needs: credentials, the data pulled on the machine, the log folder synced back. Suggested 2026-09-30.
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
- **Test other rolling windows (ADR-003 was never searched exhaustively).** Price history is fetched at `fidelity=720` (one point per 12 h), so a 3d window has 6 points and anything under 1 day is impossible without a finer re-fetch. XGBoost (r1) puts ~57% of its gain on 7d features and ~15% on 14d, almost all of it on min/max; the volatility/trend/change features get < 2% together. Try adding 3d (and maybe 30d), and dropping the unused stats; keep it only if the holdout AUC improves (ADR-024). Needs a snapshot rebuild → new dataset version + ADR. Suggested 2026-10-01.
- **Realistic trading fills.** The trading simulation assumes fills at the recorded price. Large model–market gaps are probably more common in thin markets, where that price may be a stale last trade. Needs order-book / spread history.
- **Incremental merge and leakage steps.** They rewrite the whole parquet each run (~20–40 min). Only if it becomes a problem.
- **Clean-up** (no rush):
  - `data/polymarket_ml_dataset.v2.csv`, `data/*.bak.csv`, `data/*_part[12]*.parquet`.

---

## Ongoing

- Keep ADRs up to date for any new architectural decisions
- Add new ideas and follow-ups to this file as they come up (see the rule at the top)
- Run `python scripts/smoke_test.py` after any pipeline changes
