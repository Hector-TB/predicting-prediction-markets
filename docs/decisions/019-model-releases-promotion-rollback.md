# ADR-019: Immutable Model Releases, Change Reports, Gated Promotion and Rollback

**Date:** 2026-09-27  
**Status:** Accepted; implementation plan accepted 2026-09-29 (see Amendment: one-command refresh)  
**Deciders:** Hector Thompson Baroni

---

## Context

Models will be retrained from this repo on newer data (fetch → dataset version → train). The site serves one set of models and records every live prediction (ADR-017). Without discipline, a retrain can silently replace a good model with a worse one, results from different model versions get mixed, and nobody can say why metrics moved. The user asked for a strict versioning mode: see whether and why metrics changed, and be able to roll back.

Dataset versions are already immutable (ADR-016). Models need the same.

## Decision

1. **Model releases are immutable,** numbered `r1`, `r2`, …, stored on S3 under `models/<release>/`, never overwritten.
   - **Contents:** all six model artifacts plus a `manifest.json` recording:
     - dataset version and git commit;
     - hyperparameters and feature list per model;
     - metrics, feature importance and calibration bins;
     - parent release, status (`candidate` / `promoted` / `rejected` / `retired`) and notes.
   - A copy of each manifest is committed in `models/releases/<release>.json`.
2. **A `PRODUCTION` pointer** names the release the API serves. Every stored live prediction records its release (via `model_runs`), so the live track record is always reported per release.
3. **Every release gets a generated change report** against the current production release:
   - **Data:** diff of the two dataset manifests (markets added, date range, rows, YES rate, category mix).
   - **Code:** commits and files changed between the two releases' commits.
   - **Settings:** hyperparameter and feature-list diffs.
   - **Metrics:** both releases on the **same, newest test markets** (ADR-015: shared rows, market-level CIs, vs the market price), overall and by category / lifecycle stage.
   - **Attribution:**
     - (old model, old test set) vs (old model, new test set) = change due to data or the world;
     - (old model, new test set) vs (new model, new test set) = change due to retraining.
   - **Behaviour:** how far predictions move between releases on the same markets; feature-importance shifts.
4. **Strict mode — a release is refused unless:**
   - the git working tree is clean;
   - the dataset is a published version whose local files match its manifest;
   - the smoke tests pass;
   - every model trained successfully.
5. **Promotion is gated and explicit.**
   - **Automatic checks:**
     - no statistically significant regression in ΔAUC vs the market;
     - log-loss and calibration error no worse than production beyond a small tolerance (set when implemented).
   - **If the checks pass,** promotion still requires an explicit confirmation.
   - **If they fail,** the release is marked `rejected`, with its report, and production is unchanged.
6. **Rollback** (`release.py rollback [--to rN]`) moves `PRODUCTION` to the previous promoted release (or a named one). The API reloads it without a redeploy. Nothing is deleted. Promotions and rollbacks are appended to a release log with timestamps and reasons.
7. **Hyperparameters:** routine releases retrain with the last promoted hyperparameters. Tuning is re-run only when the data has grown substantially or metrics degrade, and the report says so.

## Rationale

- Immutable releases plus a pointer make rollback instant and safe, and keep every live prediction attributable to the exact models that made it.
- Comparing releases on the same test markets is the only fair comparison: the ADR-001 split moves with each dataset version.
- The 2×2 attribution separates "the models got worse" from "the newest markets are harder", which otherwise look identical.
- Human confirmation before promotion suits a public site where model changes affect a published track record.

## Alternatives considered

- **MLflow / a model registry service** — provides tracking and stage transitions, but adds a server to run; the S3 + manifest pattern already used for data covers what's needed. Revisit if experiments multiply.
- **Automatic promotion when checks pass** — faster, but a silent model swap on a public site is exactly what this ADR prevents.
- **Overwrite-in-place with git-tracked metrics** — no rollback of artifacts; rejected.

## Consequences

- The retrain command produces a candidate release plus its report; promotion is a separate step.
- The API loads models from the `PRODUCTION` release, not from `models/*/artifacts/`.
- `model_runs` gains a release identifier; model pages and the feed can show backtest and live results per release.
- The first release (`r1`) will be the v2-trained models once ADR-018 is decided.

## Amendment (2026-09-29): prediction files move to S3 now, before releases exist

Test-set prediction CSVs (`models/*/predictions/*.csv`, 3–40 MB each) were tracked in git; every retrain would add a ~500k-line diff. They are now gitignored and stored at `s3://<bucket>/predictions/<dataset version>/<model>/<file>` with `scripts/sync_predictions.py push | pull | list`. The dataset version comes from each file's `dataset_version` column. The course paper's files (no such column) were uploaded as `v1` and remain in git history up to commit e7e44e1.

This is interim: re-pushing a changed file replaces it, and S3 bucket versioning keeps the old copy. Once releases exist, predictions belong to a release (`models/<release>/`) and are immutable, as decided above.

## Amendment (2026-09-29): one-command refresh — implementation plan

Written after the v3 models, the shared training protocol (ADR-024) and the end-date fetch (ADR-020) exist. It updates two points above that are out of date:
- a release holds **three** models (LR, XGBoost, RF), not six. The Trends variants wait for the Trends rework; SVM isn't part of the protocol;
- **`r1` is the current v3 models**, not v2.

### The command

```
python scripts/refresh.py                 # data → dataset vN+1 → candidate release rN → report
python scripts/refresh.py --resume        # continue a failed or interrupted run from its last completed stage
python scripts/refresh.py --tune          # re-run the settings search instead of reusing production's
python scripts/refresh.py --full-coverage # full coverage recount instead of the recent window
python scripts/refresh.py --no-publish    # stop after the data gates, before publishing

python scripts/release.py status | report rN | promote rN | reject rN | rollback [--to rN]
```

`refresh.py` runs the stages below in order. Each stage records its result in `logs/refresh_<run id>/state.json`, so `--resume` restarts at the first stage that didn't finish. A full run takes ~2 h, and background jobs die if VS Code quits.

| # | Stage | What it does | Stops the run if |
|---|---|---|---|
| 0 | **Preflight** | Clean git tree; local data matches the LATEST dataset manifest (`sync.py status`); `meta_checks.py` passes; production release known | any check fails |
| 1 | **Fetch** | `fetch_markets.py` incremental by end date (ADR-020) | the fetch fails, or finds 0 new markets (nothing to do) |
| 2 | **Move the split** | `recompute_split.py`: 80/20 by resolution time over all markets, so T moves forward (see *Split* below) | T would move backwards |
| 3 | **Build** | `run_pipeline.py --skip-markets --skip-trends`: snapshots for new markets only, merge, categorise new markets, Trends join (existing Trends data), leakage filter | any step fails |
| 4 | **Data gates** | `meta_checks.py`; `coverage_check.py` over recent end dates (see *Coverage*); `check_snapshots.py --compare-with <parent>`; `smoke_test.py` | any gate fails |
| 5 | **Publish** | `stage_backup.py`, then `sync.py publish vN+1 --parent vN` with an auto-generated note (counts, new T) | publish refuses |
| 6 | **Train** | LR, XGBoost, RF with production's settings (`--params`), unless `--tune` | a model fails |
| 7 | **Candidate release** | Upload to `s3://…/models/rN/`: the three artifacts (model + calibrator + threshold), test predictions, `manifest.json` (status `candidate`); commit a copy to `models/releases/rN.json` | upload fails |
| 8 | **Change report** | `models/releases/rN_report.md` vs production (see *Report*); run the promotion checks and print the verdict | — (a failed check marks the release `rejected`) |

Promotion is never part of `refresh.py`. `release.py promote rN` shows the report's verdict and asks for confirmation. It moves the `PRODUCTION` pointer (`s3://…/models/PRODUCTION`, with a git copy in `models/releases/PRODUCTION`), appends to `models/releases/log.md` and loads the release's metrics into `model_runs`. `rollback` moves the pointer back; nothing is deleted.

### Split: move T forward on every refresh

With a frozen split, new markets only ever join the test set, so retraining trains on exactly the same markets and produces the same model. Each refresh re-applies ADR-021's rule (first 80% by resolution time = train) to all markets, so T moves forward. Every training label is still known by T, and only test rows dated after the new T are scored. This re-applies ADR-021 rather than changing it.

Consequence: each dataset version has its own test set. So releases are never compared on their own reported metrics, only on the **newest version's test rows** (below), and the old model has never seen those markets.

### Coverage: recent window by default, full recount occasionally

A full `coverage_check.py` walks ~1M raw markets (~3.4 h). A market can only be missed if it closed since the previous fetch, so the default refresh checks scheduled end dates from **previous fetch − 90 days** onwards. That's three times ADR-020's 30-day look-back, so it also tests that assumption. The refresh runs the **full recount automatically when the last one is more than 30 days old** (or with `--full-coverage`), so nobody has to remember. The date of the last full recount is kept in `data/fetch_state.json`, and the manifest records which check ran. This refines ADR-020 decision 4.

### Data gate: compare with the parent version

`check_snapshots.py` gains a parent-comparison mode for incremental versions:
- markets in both versions keep identical snapshot rows, except the `split` column (moved T) and categories;
- markets only in the new version must be newly fetched;
- no market disappears unless it was removed from the meta on purpose, which must be listed in the publish note.

The existing v2 → v3 rules stay available for full rebuilds.

### Settings: reuse production's unless tuning

Each `train.py` gains `--params <json>` to skip the settings search and use the given settings. XGBoost uses the settings plus its tree count. The refresh passes production's settings from its manifest. This is ADR-019 decision 7: re-tune only when asked or when the report flags degradation. It keeps a routine XGBoost run at minutes instead of ~1.5 h. Each `train.py` also exposes `predict(artifact, df)`, so the report can run an old release on new rows.

### Report

`models/releases/rN_report.md`, built from the functions in `analysis/rescore_paper.py` and `models/common/evaluation.py`:
1. **Data:** parent vs new manifest (markets added, T, test size, YES rate, category mix).
2. **Code:** `git log` between the two releases' commits; settings and feature diffs.
3. **Metrics on the new version's test rows:** candidate vs production vs market price. Includes ΔAUC, Δlog-loss and ΔBrier with market-level CIs, both snapshot- and market-weighted (ADR-015), plus AUC by time left and by category.
4. **Attribution (2×2):** production on its own test rows vs production on the new test rows (the world changed), and production vs candidate on the new test rows (the retrain changed things).
5. **Behaviour:** how far predictions move between the two on the same rows; feature-importance shifts.

### Promotion checks

On the new version's test rows, candidate vs production:
- **AUC:** fail if the 95% CI of (candidate − production) is entirely below 0 (a significant regression);
- **log-loss:** fail if the 95% CI of (candidate − production) is entirely above 0. A fixed tolerance (e.g. +0.005) was rejected: log-loss differences have CIs about ±0.007 wide on v3, so a fixed cut-off would sit inside the noise;
- **calibration:** fail if the candidate's expected calibration error is more than 0.01 above production's;
- **vs the market:** warn (don't fail) if the candidate's ΔAUC over the market has a CI that includes 0.

Passing only makes the release promotable: `promote` still asks.

### Bootstrapping

- **`r1`** = the current v3 artifacts, packaged by `release.py init` from `models/*/artifacts/` and the v3 predictions on S3, and promoted directly: there's no production to compare against.
- **`model_runs`** gains a `release` column (migration 005); loaded runs are tagged with their release.
- **The first `refresh.py` run** produces v4 (the 737 markets) and candidate `r2`.

### Out of scope

- Serving (the API reads `PRODUCTION`; ADR-017);
- live-prediction storage;
- making the merge / leakage steps incremental (they take under an hour; revisit only if they become a problem);
- Google Trends refresh (skipped until the rework).

### Decisions from review (2026-09-29)

1. **T moves on every refresh.**
2. **Datasets are published automatically** once the data gates pass (`--no-publish` stops before).
3. **Promotion checks** as above: significance-based for AUC and log-loss; calibration error at most +0.01; the market comparison warns only.
4. **Full coverage recount** automatically when the last one is more than 30 days old.

### Later: walk-forward backtest (parked)

A single out-of-time test says a model did well in one period. A walk-forward backtest (Dhar & Stein 1998; Sobehart, Keenan & Stein 2000) tests the whole modelling approach across many periods:
- at each quarterly cutoff from 2025 Q1, train on markets resolved before it (ADR-024 protocol) and predict markets resolving in the next quarter (rows dated after the cutoff only);
- pool the predictions and score them against the market, per quarter and overall.

It answers what one window can't:
- whether the edge over the market is stable across periods or specific to July–September 2026;
- in which conditions the models fail;
- how fast a model goes stale (so how often to refresh);
- the normal quarter-to-quarter swing, to read the live track record and to check these promotion tolerances;
- a check on data not yet used for any decision.

**How:** `analysis/walk_forward.py`, reusing `train.py --params` and `predict()`. Roughly 30 min per quarter with fixed settings. Run it once, then whenever settings are re-tuned or features change, never inside routine refreshes.

**Two rules from the source:**
- never develop the model against walk-forward results, or they stop being out-of-time;
- only point-in-time inputs. Google Trends is excluded until its rework, because its scale uses later peaks.


## Related ADRs

- ADR-001: Temporal split (why test sets move between versions)
- ADR-015: Evaluation methodology (how releases are compared)
- ADR-016: Dataset versions (what a release is trained on)
- ADR-017: Live track record (why predictions must name their release)
- ADR-018: Volume feature (decided: dropped, ADR-021)
- ADR-020: Incremental fetch (coverage check scoped here)
- ADR-021: Split by resolution time (re-applied on every refresh)
- ADR-024: Shared training protocol (what each release trains with)
