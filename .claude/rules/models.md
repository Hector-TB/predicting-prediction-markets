---
paths:
  - "models/**/*.py"
---

# Model file conventions

- Import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation` — never redefine them
- Never import `roc_auc_score`, `log_loss`, `f1_score`, `average_precision_score`, or `brier_score_loss` directly in model files — they live in `models.common.evaluation`
- Always use `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost) — class imbalance is 80% NO / 20% YES
- `ROOT = Path(__file__).resolve().parent.parent.parent` for scripts two levels deep under `models/`
- Prediction CSVs must include: `market_id`, `snapshot_timestamp`, `outcome`, `dataset_version`. They are gitignored; push them to S3 with `scripts/sync_predictions.py`
- Fit only on `split == "train"` and score only on `split == "test"`; `test_pre_cutoff` rows are never used (ADR-021)
- Follow the shared training protocol (ADR-024) from `models.common.training`: `load_dataset`, `split_holdout` (newest 20% of train markets by resolution time), `market_weights` as `sample_weight`, `fit_calibrator`. Don't add per-model variants
- Never tune anything on the test set: settings, calibration and the decision threshold all come from the holdout (ADR-021, ADR-024)
- Any cross-validation must group by `market_id` (`StratifiedGroupKFold`), so one market's snapshots never sit in two folds
- Never use `total_volume` / `log_volume` as features: they are the final lifetime volume (ADR-018)
