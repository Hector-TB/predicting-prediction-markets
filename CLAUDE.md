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

## Current state (as of 2026-09-29)

Dataset versions on S3 (ADR-016, manifests in `data/manifests/`): **v1** = the course paper's data; **v2** = full re-fetch + leakage filter + split by resolution time (ADR-013/014/021); **v3 (LATEST)** = v2 without the 14-day pre-close cutoff, metadata matched to Gamma (ADR-022/023). Local `data/` holds v3.

- v3: 45,130 markets; 2,534,852 clean snapshots (train 2,038,895 / 23,120 markets; test 196,977 / 4,924; `test_pre_cutoff` 298,980 unused); cutoff T = 2026-07-01 07:11 UTC. Coverage check: 0 markets missed; full quality check passed
- LR, XGBoost and RF trained on v3 under one protocol (ADR-024, `models/common/training.py`); re-score in `docs/paper/rescore_v3_data.md`. README results are still the paper's (v1). v2 won't be trained (superseded by v3)
- Gates before any publish: `meta_checks.py` (in the pipeline), `scripts/coverage_check.py`, `scripts/check_snapshots.py`
- Leftovers to delete when convenient (user said no rush): `data/polymarket_ml_dataset.v2.csv` (on S3 staging), `data/*.bak.csv`, `data/*_part[12]*.parquet`, `s3://…/staging/2026-09-27/` and `staging/2026-09-28-v3/`
- Supabase (free tier): still holds v1. `db/load_parquet.py` now upserts. Re-check tables after any unpause before assuming data loss
- S3 bucket versioning on; work is on branch `pipeline-leakage-and-categories` (pushed, not merged to `main`)
- Machine: 8 GB RAM, 2 physical cores; VS Code runs from a translocated path (move it to Applications). Background jobs started from Claude die if VS Code quits
- No API, no frontend yet

## Next session — start here (in order)

1. ~~**Train on v3**~~ ✓ 2026-09-29: LR, XGBoost, RF (Trends variants skipped until the Trends rework, ROADMAP 4c). Predictions on S3 (`predictions/v3/`).
2. **Re-score** ✓ `docs/paper/rescore_v3_data.md`: XGBoost/RF beat the market (ΔAUC ≈ +0.014, 95% CI ≈ [+0.009, +0.020]; LR doesn't). The edge is entirely ≥ 30 days before close; in the last 30 days models ≈ market. README results updated (v3 table first, the paper's v1 table kept as published).
3. **Refresh Supabase:** `python3 db/load_parquet.py`.
4. **Merge** `pipeline-leakage-and-categories` → `main` (or open a PR).
5. **Then build:** the one-command refresh + model releases (ROADMAP 4b, ADR-019) with the three checks as gates; the fixed incremental fetch (ADR-020; don't run a plain incremental `fetch_markets.py` before it); then the site prerequisites (ROADMAP 5) and the API/site. Separate tasks: Trends rework, volume to date (ROADMAP 4c). 692 markets closed after the 25 Sep fetch will come in with the next fetch.

## Key conventions

**Market metadata (ADR-023):** every field except `category`/`split` must equal Gamma's — never estimate, back-fill or placeholder a value; drop markets that can't be verified. `python data_collection_pipeline/meta_checks.py` must pass.

**Path resolution:** `ROOT = Path(__file__).resolve().parent` (or `.parent.parent` as needed) — never hardcode relative paths.

**Metrics:** primary = AUC-ROC and log-loss. Market price baseline = `price_at_snapshot` scored on the same test set — get current numbers from `python scripts/print_metrics.py`, never a hard-coded value (the old 0.964 / 0.171 did not match the clean data; ADR-014 changes it again). Do not report raw accuracy (78% NO class imbalance makes it misleading).

**Class imbalance:** always `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost). No exceptions.

**Training protocol (ADR-024):** every model uses `models/common/training.py`: holdout = newest 20% of train markets by resolution time; `market_weights` (each market counts once); settings by holdout AUC; isotonic calibration + threshold from the holdout. Don't add per-model variants.

**Train/test split:** market-level, by resolution time — first 80% resolved = train; test rows dated before the cutoff are `test_pre_cutoff` (unused). Never mix snapshots from the same market across splits; never tune thresholds or anything else on the test set. See ADR-021.

**Shared evaluation code:** import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation` — never redefine them.

**Model comparisons (ADR-015):** score every model and the market baseline on the same test rows; get CIs by resampling markets (`bootstrap_auc_diff(..., groups=market_id)`); in trading simulations a NO trade costs `1 − p`.

**No bare `print()` in new code** — use Python's `logging` module.

---

## What NOT to do

- Do not modify `data/*.parquet` directly — they live on S3, manage with `data/sync.py`
- Prediction CSVs (`models/*/predictions/*.csv`) are gitignored and live on S3: after a training run, `python scripts/sync_predictions.py push`; to fetch, `pull --version v3`
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
- ADR-021: leak-free evaluation — split by resolution time + pre-cutoff test rows dropped, thresholds from held-out train data, `log_volume` dropped
- ADR-022: no 14-day pre-close cutoff from v3 (price rule only); full snapshot rebuild with retries + settings guard
- ADR-023: meta must match Gamma — no estimated/placeholder values; fresh fetch wins on merge (category/split kept); `meta_checks.py` blocks pipeline/build/publish; repair with `scripts/refresh_meta.py`
- ADR-024: one training protocol for all models — time-ordered holdout, one weight per market, calibration + threshold from the holdout (supersedes ADR-021 decisions 2–3)
