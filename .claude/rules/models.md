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
