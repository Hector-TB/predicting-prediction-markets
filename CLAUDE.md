# CLAUDE.md — Predicting Prediction Markets

<!-- When compacting: preserve the active data split (train=80%/test=20% temporal at market level), current ADR count (ADR-001 through ADR-012), and any files modified this session. -->

This file is read by Claude Code at the start of every session. Keep it under 200 lines.

---

## What this project is

ML models on historical Polymarket binary prediction markets to forecast YES/NO outcomes.
Originally NYU DS-GA 1003 (team of 3). Now being productionized into a full-stack app.

**Target architecture:** data pipeline → PostgreSQL (Supabase) → FastAPI → React → model serving.

---

## Current state (as of 2026-09-19)

- Phase: productionization — data pipeline hardened, working toward live scoring
- Data: parquet files on S3 (`python data/sync.py pull` to fetch); pipeline runner at `data_collection_pipeline/run_pipeline.py`
- Models trained: logistic regression, XGBoost, SVM (±trends), random forest (train.py done, not yet retrained on latest data)
- Database: Supabase (free tier); schema in `db/migrations/001_initial_schema.sql`; 20,948 markets + 9 model_runs loaded
- No API, no frontend yet

---

## Key conventions

**Path resolution:** `ROOT = Path(__file__).resolve().parent` (or `.parent.parent` as needed) — never hardcode relative paths.

**Metrics:** primary = AUC-ROC and log-loss. Market price baseline = AUC 0.964 / log-loss 0.171. Do not report raw accuracy (78% NO class imbalance makes it misleading).

**Class imbalance:** always `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost). No exceptions.

**Train/test split:** market-level temporal — oldest 80% train, newest 20% test. Never mix snapshots from the same market across splits. See ADR-001.

**Shared evaluation code:** import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation` — never redefine them.

**No bare `print()` in new code** — use Python's `logging` module.

---

## What NOT to do

- Do not modify `data/*.parquet` directly — they live on S3, manage with `data/sync.py`
- Do not add model-specific evaluation functions — put them in `models/common/evaluation.py`
- Do not amend commits that have been pushed
- Do not change the train/test split logic without a new ADR

---

## Database

**Split: parquet for training, Postgres for operations (ADR-011).**

- Training scripts read parquet directly — never query the database
- DB holds: `markets` (metadata), `trends`, `model_runs` (with metrics JSONB)
- DB will hold `snapshots` + `predictions` for live markets only (not backfilled)
- To populate: `python db/load_parquet.py` (runs in seconds)
- Schema includes metrics + hyperparams JSONB on model_runs (004 is already in initial schema)

---

## Skills (invoke with /<name>)

- `/evaluate` — print current test metrics for all models vs baseline
- `/sync-and-train` — pull from S3 + run training pipeline
- `/add-model <name>` — scaffold a new model directory

---

## Architecture decisions (docs/decisions/)

- ADR-001: temporal market-level 80/20 train/test split
- ADR-002: snapshot window design
- ADR-003: rolling window selection (7d + 14d)
- ADR-004: outcome threshold
- ADR-005: leakage fix (remove post-resolution snapshots)
- ADR-006: SVM subsampling
- ADR-007: class imbalance handling
- ADR-008: market filters
- ADR-009: Google Trends enrichment
- ADR-010: database design (Supabase, 5-table schema)
- ADR-011: offline/online split (parquet vs DB)
- ADR-012: S3 + DuckDB data lake
