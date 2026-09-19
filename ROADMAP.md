# Roadmap

Ordered by priority. See `docs/decisions/` for the reasoning behind architectural choices.

---

## Now — unblock the foundation

### 1. Apply pending DB migration + populate DB
- Wait for Supabase to recover from hot standby
- Apply `db/migrations/004_add_model_run_metrics.sql` (adds `metrics` JSONB to `model_runs`)
- Run `python db/load_parquet.py` (markets already loaded; adds trends + model_runs with metrics)

### 2. Audit and harden the data refresh pipeline
The pipeline scripts were written for a one-time run. Before treating them as operational, verify:
- `update_markets.py` correctly fetches only markets newer than the last run
- `build_snapshots.py` checkpoint/resume works correctly on incremental runs
- `categorize_markets.py` skips already-categorised markets
- `fetch_category_trends.py` handles date ranges that extend the existing trends data
- End-to-end: run the full pipeline from step 1 → step 6 on fresh data, push to S3

### 3. Write `train.py` for random forest
`models/random_forest/` and `models/random_forest_trends/` are notebook-only.
Need `train.py` files so both models are included in `/sync-and-train` and `/evaluate`.
Follow the pattern in `models/gradient_boosting/train.py`.

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
