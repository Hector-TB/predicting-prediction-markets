# ADR-019: Immutable Model Releases, Change Reports, Gated Promotion and Rollback

**Date:** 2026-09-27  
**Status:** Accepted — to be implemented with the one-command refresh after the first v2 run  
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

## Related ADRs

- ADR-001: Temporal split (why test sets move between versions)
- ADR-015: Evaluation methodology (how releases are compared)
- ADR-016: Dataset versions (what a release is trained on)
- ADR-017: Live track record (why predictions must name their release)
- ADR-018: Volume feature (to decide before r1)
