# CLAUDE.md — Predicting Prediction Markets

<!-- When compacting: preserve the active data split (80/20 at market level by resolution time, ADR-021), current ADR count (ADR-001 through ADR-023), and any files modified this session. -->

This file is read by Claude Code at the start of every session. Keep it under 200 lines.

---

## What this project is

ML models on historical Polymarket binary prediction markets to forecast YES/NO outcomes.
Originally NYU DS-GA 1003 (team of 3). Now being productionized into a full-stack app.
The course paper (research questions, methods, reported results) is `docs/paper/paper.tex`; see `docs/paper/README.md` for how its numbers relate to the current data.

**Target architecture:** data pipeline → PostgreSQL (Supabase) → FastAPI → React → model serving.

---

## Current state (as of 2026-09-27, night)

Dataset versions: **v1** = the course-project dataset the paper used (on S3, `datasets/v1/`, frozen); **v2** = the ADR-013/014 rebuild (45,143 markets fetched, not yet published).

- Snapshot rebuild finished 2026-09-27 ~22:20: `data/polymarket_ml_dataset.csv` has 2,788,184 rows (COMPLETE, no errors). Backed up to `s3://<bucket>/staging/2026-09-27/` (CSV, meta, closed-times cache, fetch cache, build logs — all sizes verified; log `logs/stage_backup.log`). This is a backup, **not** a dataset version
- All 45,143 markets are LLM-categorised (`data/polymarket_markets_meta.csv`); backups `data/*.pre_categorize*.bak.csv` can be deleted after v2 is published
- Supabase (free tier; was paused, restored 2026-09-27): still holds v1 — 20,948 markets, 1,458 trends, 9 model_runs, empty snapshots/predictions. Re-check tables after any unpause before assuming data loss
- S3 bucket versioning enabled 2026-09-27; `sync.py push` removed (ADR-016)
- Work is on branch `pipeline-leakage-and-categories` (pushed to GitHub, not merged to `main`)
- Machine: 8 GB RAM, disk often < 5 GB free — restart before heavy steps; run models one at a time
- No API, no frontend yet

## Next session — start here (in order)

State at end of 2026-09-28: v2 published (S3 `datasets/v2/`, LATEST). The v3 rebuild (ADR-022, meta refreshed per ADR-023) ran 12:05–~18:00. `scripts/stage_backup.py` (log `logs/stage_backup_v3.log`) waited for the pipeline and the coverage check, then backed everything up to `s3://<bucket>/staging/2026-09-28-v3/` with size checks. v3 is **not published yet**.

1. **Check last night's jobs:** end of `logs/pipeline_v3_20260928_1205.log` (`PIPELINE COMPLETE`?), `logs/stage_backup_v3.log` (`DONE`, no `SIZE MISMATCH`), and the latest `logs/coverage_run_*.log`.
2. **Coverage result (ADR-020):** if any markets were `MISSED` (list in `logs/coverage_*.csv`), work out why and fetch them *before* publishing v3. `new` = closed after the 2026-09-25 fetch (the next fetch adds them); `out of scope` = Gamma startDate before 2023.
3. **Full quality check on v3** (same checks as the mid-build ones: invariants, window vs meta/closedTime, v2 comparison on shared rows), then the clean-parquet split summary (train / test / test_pre_cutoff).
4. **Publish v3:** `python3 data/sync.py publish v3 --parent v2 --notes "…"` (dry run first); commit the manifest. Then delete `data/polymarket_ml_dataset.v2.csv` and `staging/2026-09-28-v3/` (optional).
5. **Train on v2 and v3**, one at a time: LR and XGBoost (each ± `--trends`), RF, RF + Trends. Only apply memory fixes if a run is killed (exit 137).
6. **Re-score:** `python3 analysis/rescore_paper.py` on v2 and v3; compare, including by `days_before_close` bucket. Then update the README results (neutral tone, see memory).
7. **Refresh Supabase:** `python3 db/load_parquet.py` (now upserts).
8. **Merge** `pipeline-leakage-and-categories` → `main` (or open a PR).
9. **Then build:** the one-command refresh + model releases (ROADMAP 4b, ADR-019), with `coverage_check.py` and `meta_checks.py` as gates; the fixed incremental fetch (ADR-020; don't run a plain incremental `fetch_markets.py` before it); then the site prerequisites (ROADMAP 5) and the API/site. Separate tasks: Trends rework, volume to date (ROADMAP 4c).

## Key conventions

**Market metadata (ADR-023):** every field except `category`/`split` must equal Gamma's — never estimate, back-fill or placeholder a value; drop markets that can't be verified. `python data_collection_pipeline/meta_checks.py` must pass.

**Path resolution:** `ROOT = Path(__file__).resolve().parent` (or `.parent.parent` as needed) — never hardcode relative paths.

**Metrics:** primary = AUC-ROC and log-loss. Market price baseline = `price_at_snapshot` scored on the same test set — get current numbers from `python scripts/print_metrics.py`, never a hard-coded value (the old 0.964 / 0.171 did not match the clean data; ADR-014 changes it again). Do not report raw accuracy (78% NO class imbalance makes it misleading).

**Class imbalance:** always `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost). No exceptions.

**Train/test split:** market-level, by resolution time — first 80% resolved = train; test rows dated before the cutoff are `test_pre_cutoff` (unused). Never mix snapshots from the same market across splits; never tune thresholds or anything else on the test set. See ADR-021.

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

- ADR-001: temporal market-level 80/20 train/test split (split rule superseded by ADR-021)
- ADR-002: snapshot window design (14-day cutoff superseded by ADR-022 from v3)
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
- ADR-017: on-demand prediction via search + stored live track record (design: `docs/design/search-and-predict.md`)
- ADR-018: volume features used final lifetime volume (look-ahead) — dropped (ADR-021)
- ADR-019: immutable model releases (rN) with change reports, gated promotion, one-command rollback
- ADR-020: incremental fetch by scheduled end date (≥ last fetch − 30d, no upper bound) — current start-date fetch misses long-running markets; pre-publish coverage check
- ADR-021: leak-free evaluation — split by resolution time + pre-cutoff test rows dropped, thresholds from held-out train data, grouped CV, `log_volume` dropped
- ADR-022: no 14-day pre-close cutoff from v3 (price rule only); full snapshot rebuild with retries + settings guard
- ADR-023: meta must match Gamma — no estimated/placeholder values; fresh fetch wins on merge (category/split kept); `meta_checks.py` blocks pipeline/build/publish; repair with `scripts/refresh_meta.py`
