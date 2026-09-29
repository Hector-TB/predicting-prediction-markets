---
paths:
  - "models/**/*.py"
---

# Model file conventions

- Import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation` — never redefine them
- Never import `roc_auc_score`, `log_loss`, `f1_score`, `average_precision_score`, or `brier_score_loss` directly in model files — they live in `models.common.evaluation`
- Always use `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost) — class imbalance is 80% NO / 20% YES
- `ROOT = Path(__file__).resolve().parent.parent.parent` for scripts two levels deep under `models/`
- Prediction CSVs must include: `market_id`, `snapshot_timestamp`, `outcome`
- Fit only on `split == "train"` and score only on `split == "test"`; `test_pre_cutoff` rows are never used (ADR-021)
- Never tune anything on the test set: decision thresholds come from the calibration markets or out-of-fold train predictions (ADR-021)
- Cross-validation must group by `market_id` (`StratifiedGroupKFold`), so one market's snapshots never sit in two folds
- Never use `total_volume` / `log_volume` as features: they are the final lifetime volume (ADR-018)
