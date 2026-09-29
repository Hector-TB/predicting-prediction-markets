# Roadmap

Ordered by priority. See `docs/decisions/` for the reasoning behind architectural choices.

---

## Now — unblock the foundation

### 1. ~~Apply pending DB migration + populate DB~~ ✓ DONE
Schema in `db/migrations/001_initial_schema.sql`. All 5 tables created.
20,948 markets · 1,458 trends · 9 model_runs loaded via `db/load_parquet.py`.

### 2. ~~Audit and harden the data refresh pipeline~~ ✓ DONE
All 5 issues fixed: incremental fetch in `fetch_markets.py` (frozen split); dynamic
timeframe + append in `fetch_category_trends.py`; absolute paths in `build_trend_features.py`;
parquet output in `fix_dataset.py`; dynamic `HARD_CUTOFF` + single-file input in `fix_leakage.py`.
- 2026-09-25–27: keyset pagination + full re-fetch (ADR-013, 45,143 markets); working
  closedTime lookup + per-row settled-price filter (ADR-014); LLM categories keyed by id
  (all markets re-categorised); pipeline streams in batches to fit 8 GB of RAM
- 2026-09-28: pre-v2 review fixed six pipeline bugs (Trends scale/look-ahead, runner step
  skipping, DB loader upserts, …). **v2 published.** Leak-free evaluation (ADR-021): split by
  resolution time, `log_volume` dropped, thresholds and CV off the test set.
- 2026-09-28: **v3 published**: no 14-day pre-close cutoff (ADR-022); metadata matched to Gamma after
  14,886 estimated dates were found (ADR-023, checks now block build and publish). Parallel
  snapshot build (about 5 h). Coverage check (ADR-020): 0 markets missed. Full quality check
  (`scripts/check_snapshots.py`) passed.

### 3. ~~Write `train.py` for random forest~~ ✓ DONE
`models/random_forest/train.py` and `models/random_forest_trends/train.py` written.
Both follow the shared training protocol (ADR-024) since 2026-09-29.

### 4. Retrain all models on v3 and re-score the paper
- ~~Unify the training protocol~~ ✓ 2026-09-29 (ADR-024): time-ordered holdout, one weight per market,
  LR now calibrated. v2 dropped from the plan (superseded by v3)
- ~~Train LR, XGBoost and RF on v3~~ ✓ 2026-09-29. Predictions on S3 (`predictions/v3/`). Trends variants
  (LR/XGBoost `--trends`, RF + Trends) skipped until the Trends rework below
- ~~Re-score~~ ✓ `docs/paper/rescore_v3_data.md`, README results updated: XGBoost/RF beat the market
  (+0.014 AUC, CI above zero), with all of the gain ≥ 30 days before close; LR is level with the market
- ~~Extra re-score metrics~~ ✓ CIs on log-loss/Brier differences, market-weighted scores, time left × duration,
  trading with costs and thresholds
- Refresh the DB (`python db/load_parquet.py`)

---

### 4b. One-command refresh with strict model releases (ADR-019)
Turn the manual steps into one command (`/sync-and-train`):
fetch (by scheduled end date, ADR-020) → snapshots → pipeline → `meta_checks.py` + `coverage_check.py` +
`check_snapshots.py` → S3 staging backup (`stage_backup.py`) → publish dataset vN → train → candidate release rN with a
"what changed and why" report vs production → gated, explicit promotion; `rollback` restores
the previous release.

### 4c. Feature research (separate tasks)
- ~~Dataset v3 (ADR-022)~~ ✓ published 2026-09-28. No v2 model comparison: the late-row question is answered within v3.
- **Rework Google Trends.** The current setup is weak: category-level keywords, one 0–100 scale
  over the whole period (the scale depends on later peaks), weekly granularity. Revisit before relying on it
  (ADR-009 amendment lists the known issues).
- **Volume to date.** Rebuild volume as cumulative traded volume up to each snapshot from
  `data-api.polymarket.com/trades` (ADR-018 option 2). That endpoint caps at ~10.5k trades per market, so large
  markets need another source (e.g. the orderbook subgraph). New ADR + dataset version.

## Next — search-and-predict with a live track record (ADR-017, `docs/design/search-and-predict.md`)

### 5. Prerequisites
- ~~Decide and apply ADR-018~~ ✓ `log_volume` dropped (ADR-021)
- Refactor the per-snapshot feature code in `build_snapshots.py` into a shared function used by
  training and the API; test it reproduces training snapshots exactly
- DB migration: full feature vector + warnings on `snapshots`; resolution status on `markets`

### 6. FastAPI backend
- `GET /resolve?q=` — URL / slug / event / condition ID / name → market(s)
- `GET /predict?market_id=` — features as of now → all six models' probabilities + gap vs price,
  with warnings for out-of-distribution markets (non-binary rejected); stores every lookup
- `GET /track-record` — backtest metrics + live metrics over resolved lookups
- Daily resolution job: record outcomes of looked-up markets (ADR-004 rule)

---

## Later — frontend

### 7. React frontend (public portfolio site)
Search → market detail page (all six models + gap, prediction history, outcome), recent
predictions feed (live track record), model pages (importance, calibration, backtest vs live),
track-record comparison, research page. Blocked on: API.

---

## Ongoing

- Keep ADRs up to date for any new architectural decisions
- Run `python scripts/smoke_test.py` after any pipeline changes
