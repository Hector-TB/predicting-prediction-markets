# ADR-011: Offline/Online Split — Parquet for Training, Postgres for Operations

**Date:** 2026-09-19  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  

---

## Context

After designing the Supabase schema (ADR-010), the first attempt to load historical data into the database revealed a fundamental sizing problem: the `snapshots` table alone contains 1.4 million rows × 30+ numeric columns, which is several GB of data. This exceeds the Supabase Pro tier's comfortable operating range and makes the migration extremely slow and fragile (SSL timeout at 69% during initial load attempt).

More importantly, loading historical training data into a transactional database conflates two very different concerns:

1. **Offline training** — batch computation on a fixed historical dataset
2. **Online operations** — serving live predictions, tracking model performance, browsing markets

## Decision

**Parquet files are the training data store. Postgres is the operational/live layer.**

Concretely:

- `data/polymarket_ml_dataset_clean.parquet` and related files stay as the canonical training dataset, read directly by training scripts via pandas/pyarrow.
- The `snapshots` and historical `predictions` tables in Postgres are **not backfilled** with the 1.4M historical rows.
- `db/load_parquet.py` loads only three lightweight tables:
  - `markets` — ~25k rows of market metadata (~5 MB)
  - `trends` — ~1.5k rows of Google Trends weekly aggregates (tiny)
  - `model_runs` — one row per model variant with computed metrics in a JSONB column (tiny)
- The `snapshots` and `predictions` tables exist in the schema and will be populated by the **live scoring pipeline** (markets currently open, predictions at the current moment). They will not hold historical data.

## Rationale

**Training scripts don't need a database.** Pandas reads parquet directly and is already the bottleneck (feature engineering, cross-validation). Replacing parquet reads with database queries adds latency, network dependency, and operational complexity for zero benefit.

**The DB serves the app, not the models.** The value of Postgres is: browsing live markets, showing live predictions, tracking model versions, and serving a frontend. None of this requires 1.4M historical rows.

**Free tier is viable.** With only metadata + trends + model registry, the database is well under 500 MB — within Supabase's free tier. No ongoing hosting cost.

**Model metrics belong in the registry.** Rather than storing per-snapshot predictions historically, aggregate metrics (AUC-ROC, PR-AUC, log-loss, Brier, F1, optimal threshold) are computed from the prediction CSVs at migration time and stored as JSONB in `model_runs.metrics`. This gives the frontend what it needs (leaderboard, model comparison) without 1.4M prediction rows.

## Alternatives considered

- **Load everything into Postgres** — rejected; multi-GB load is slow, expensive, and the data never changes (it's a fixed historical dataset)
- **Keep parquet for training, use DuckDB as a local analytics layer** — interesting future option; DuckDB can query parquet directly with SQL syntax. Deferred.
- **TimescaleDB** — rejected (see ADR-010); overkill at this scale

## Schema changes required

Migration `004_add_model_run_metrics.sql`:
```sql
ALTER TABLE model_runs
  ADD COLUMN IF NOT EXISTS metrics     JSONB,
  ADD COLUMN IF NOT EXISTS hyperparams JSONB;
```

## Consequences

- `db/load_parquet.py` completes in seconds (was minutes/hours).
- Training scripts remain unchanged — they still read parquet directly.
- The `snapshots` table will grow organically as live markets are scored (a few hundred rows/day max, not millions).
- If historical per-snapshot predictions are ever needed in the DB (e.g., for interactive exploration), they can be loaded on demand with `--load-historical` flag added to `load_parquet.py`.
- SVM models remain excluded — their prediction CSVs lack standard probability outputs and their integration requires a separate refactor.

## Related ADRs

- ADR-010: Database design and schema
- ADR-001: Train/test split (temporal, market-level)
