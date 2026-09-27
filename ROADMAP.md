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
- IN PROGRESS: snapshot rebuild for the new markets, then `run_pipeline.py --skip-markets
  --skip-snapshots` + `python data/sync.py publish v2 --parent v1` (do before step 4)

### 3. ~~Write `train.py` for random forest~~ ✓ DONE
`models/random_forest/train.py` and `models/random_forest_trends/train.py` written.
Both follow the gradient_boosting pattern: combined sample weights, 5-fold CV, isotonic calibration.

### 4. Retrain all models on latest data and re-score the paper
After the pipeline refresh (step 2) is confirmed clean:
- Train LR and XGBoost with and without `--trends`, RF and RF + Trends (`/sync-and-train`)
- Run `python analysis/rescore_paper.py --out docs/paper/rescore_v2_data.md` — does the
  paper's RQ1 gain survive ADR-014? (evaluation rules: ADR-015)
- Run `/evaluate` and compare new vs old metrics
- Push updated predictions to S3; refresh the DB (`python db/load_parquet.py`)

---

## Next — search-and-predict with a live track record (ADR-017, `docs/design/search-and-predict.md`)

### 5. Prerequisites
- Decide and apply ADR-018 (volume features use final lifetime volume — look-ahead); retrain
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
Search / predict page (all six models + gap, headline model first), model track record
(backtest + live), research page. Blocked on: API.

---

## Ongoing

- Keep ADRs up to date for any new architectural decisions
- Run `python scripts/smoke_test.py` after any pipeline changes
