# ADR-010: Database Design — PostgreSQL via Supabase

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  

---

## Context

The project was built entirely on flat files (parquet, CSV). To productionize into a full-stack app — with live market scoring, a web UI, and model tracking — we need a relational database.

## Decision

**PostgreSQL** hosted on **Supabase** (Pro tier).

Five tables: `markets`, `snapshots`, `trends`, `model_runs`, `predictions`.

## Rationale

**Why PostgreSQL:** Structured relational data, strong Python ecosystem (SQLAlchemy + Alembic + psycopg2), excellent analytics performance with proper indexing, JSONB for flexible columns (hyperparams, metrics), battle-tested.

**Why Supabase:** Managed Postgres removes ops burden; built-in table browser useful during development; auto-generated REST API (PostgREST) reduces FastAPI boilerplate for simple reads; auth layer available when needed. Pro tier ($25/mo) required — the snapshots table alone (~1.4M rows × 30 numeric columns) exceeds the free 500MB limit.

**Why not TimescaleDB:** Snapshots table is ~1.4M rows and grows at ~70 rows/market/day for live markets. Standard Postgres with a (market_id, snapshot_timestamp DESC) index is sufficient at this scale. Revisit if ingestion volume grows to tens of millions.

## Schema decisions

**`trends` is normalized** (separate table, joined at query/feature-engineering time) rather than baked into snapshot rows. This means trend data can be updated without touching 1.4M snapshot rows.

**Predictions stored at snapshot level**, not market level. Each snapshot is the feature vector at time T; the prediction at that snapshot is the model's probability estimate at that moment. This is the core of live market scoring: "given the market state right now, what does the model say?" Per-market aggregate predictions are computed at query time from the most recent snapshot.

**`outcome` and `split` are nullable on `markets`** to support live/open markets from day one. `outcome = NULL` means the market hasn't resolved yet. `split = NULL` means it's a live market outside the historical train/test set.

**`model_runs` is the model registry.** Each training run gets a row with git SHA, dataset tag, hyperparams (JSONB), and metrics (JSONB). Predictions reference a specific model_run, so you always know which model version produced which prediction.

**`ON DELETE CASCADE`** on snapshots→markets and predictions→model_runs/snapshots, so deleting a market cleans up its snapshots and predictions automatically.

## Alternatives considered

- **SQLite** — not suitable for concurrent web access; no hosted option
- **MongoDB** — no benefit for structured tabular data; adds schema-less complexity
- **Neon** — 3GB free tier; rejected in favour of Supabase (user familiarity)
- **DuckDB** — excellent for analytics but not a production OLTP database; could be used as a local analytics layer alongside Postgres in future

## Assumptions

1. 1.4M existing snapshot rows + growing live data stays under Supabase Pro limits (8GB)
2. Standard btree indexes on (market_id, snapshot_timestamp) are sufficient for app query patterns
3. Supabase PostgREST can serve simple read-only API needs; FastAPI handles complex logic

## Migration plan

1. Schema created via Supabase MCP migrations (this ADR)
2. Existing parquet data loaded via one-time `db/load_parquet.py` script
3. Existing model prediction CSVs loaded into `model_runs` + `predictions`
4. Future pipeline writes directly to Postgres

## Consequences

- All pipeline scripts will need a `DATABASE_URL` env var going forward
- The parquet files in `data/` remain as source-of-truth backups but are no longer the live dataset
- Row-level security (RLS) should be enabled before any public-facing API is deployed

## Related ADRs

- ADR-001 through ADR-009: decisions that shaped the schema column choices
