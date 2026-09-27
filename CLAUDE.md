# CLAUDE.md — Predicting Prediction Markets

<!-- When compacting: preserve the active data split (train=80%/test=20% temporal at market level), current ADR count (ADR-001 through ADR-016), and any files modified this session. -->

This file is read by Claude Code at the start of every session. Keep it under 200 lines.

---

## What this project is

ML models on historical Polymarket binary prediction markets to forecast YES/NO outcomes.
Originally NYU DS-GA 1003 (team of 3). Now being productionized into a full-stack app.
The course paper (research questions, methods, reported results) is `docs/paper/paper.tex`; see `docs/paper/README.md` for how its numbers relate to the current data.

**Target architecture:** data pipeline → PostgreSQL (Supabase) → FastAPI → React → model serving.

---

## Current state (as of 2026-09-27)

Dataset versions: **v1** = the course-project dataset the paper used (20,948 markets, still on S3); **v2** = the ADR-013/014 rebuild (45,143 markets fetched, in progress).

- Phase: productionization — rebuilding the dataset, then retraining and re-scoring the paper's results
- Data: the ADR-013 full re-fetch grew the market list to 45,143 (all LLM-categorised); `build_snapshots.py` is rebuilding snapshots for the ~20k new markets. After it: `run_pipeline.py --skip-markets --skip-snapshots`, then `python data/sync.py publish v2 --parent v1 --notes "…"`. S3 holds v1 (immutable, `datasets/v1/`); datasets are versioned per ADR-016 — never overwrite, publish a new version
- Pipeline steps stream in batches (`data_collection_pipeline/stream_parquet.py`), so they fit in 8 GB of RAM
- Models: LR, XGBoost (`--trends` for the Trends variant), RF, RF + Trends, SVM — all still trained on the v1 dataset; retrain after the rebuild, then `python analysis/rescore_paper.py`
- Course paper results: RQ1 gains reproduce and are significant with market-level CIs; the paper's trading ROI is overstated (ADR-015, `docs/paper/README.md`)
- Database: Supabase (free tier); schema in `db/migrations/001_initial_schema.sql`; 20,948 markets + 9 model_runs loaded (not yet refreshed)
- Work in progress is on branch `pipeline-leakage-and-categories`
- No API, no frontend yet

---

## Key conventions

**Path resolution:** `ROOT = Path(__file__).resolve().parent` (or `.parent.parent` as needed) — never hardcode relative paths.

**Metrics:** primary = AUC-ROC and log-loss. Market price baseline = `price_at_snapshot` scored on the same test set — get current numbers from `python scripts/print_metrics.py`, never a hard-coded value (the old 0.964 / 0.171 did not match the clean data; ADR-014 changes it again). Do not report raw accuracy (78% NO class imbalance makes it misleading).

**Class imbalance:** always `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost). No exceptions.

**Train/test split:** market-level temporal — oldest 80% train, newest 20% test. Never mix snapshots from the same market across splits. See ADR-001.

**Shared evaluation code:** import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation` — never redefine them.

**Model comparisons (ADR-015):** score every model and the market baseline on the same test rows; get CIs by resampling markets (`bootstrap_auc_diff(..., groups=market_id)`); in trading simulations a NO trade costs `1 − p`.

**No bare `print()` in new code** — use Python's `logging` module.

---

## What NOT to do

- Do not modify `data/*.parquet` directly — they live on S3, manage with `data/sync.py`
- Never overwrite or delete a published dataset version on S3 — publish a new one (ADR-016)
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
- ADR-013: keyset pagination for market fetch + full re-fetch (split recomputed)
- ADR-014: leakage filter — working closedTime lookup + per-row settled-price rule
- ADR-015: evaluation methodology — shared test rows, market-level bootstrap CIs, NO bets cost 1−p
- ADR-016: immutable dataset versions on S3 (`datasets/vN/` + manifest); `sync.py publish/pull`
