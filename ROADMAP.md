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
- TODO: end-to-end run on fresh data + push to S3 (do before step 4)

### 3. ~~Write `train.py` for random forest~~ ✓ DONE
`models/random_forest/train.py` and `models/random_forest_trends/train.py` written.
Both follow the gradient_boosting pattern: combined sample weights, 5-fold CV, isotonic calibration.

### 4. Retrain all models on latest data
After the pipeline refresh (step 2) is confirmed clean:
- Run `/sync-and-train`
- Run `/evaluate` and compare new vs old metrics
- Push updated predictions to S3

---

## Next — live scoring

### 5. Live scoring pipeline
Fetch currently open Polymarket markets, run each through trained models, write predictions to DB.
This is the core of "what does the model say about this market right now?"
- Extend `fetch_markets.py` to also fetch open (unresolved) markets
- Build snapshots for open markets (current price + rolling features)
- Run models on those snapshots
- Write results to `snapshots` + `predictions` tables in Supabase

### 6. FastAPI backend
Serve the DB to a frontend:
- `GET /markets` — list live markets with model predictions vs market price
- `GET /markets/{id}` — prediction history for a single market
- `GET /models` — model registry + metrics

---

## Later — frontend

### 7. React frontend
Browse live markets, visualize predictions vs market price over time, surface trading edges.
Blocked on: live scoring pipeline + API.

---

## Ongoing

- Keep ADRs up to date for any new architectural decisions
- Run `python scripts/smoke_test.py` after any pipeline changes
