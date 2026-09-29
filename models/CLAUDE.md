# Models

Shared evaluation utilities live in `models/common/evaluation.py`.
Import from there — do not redefine `evaluate`, `check_calibration`, or `find_optimal_threshold`.

Every model follows the shared training protocol in `models/common/training.py` (ADR-024): `load_dataset`,
`split_holdout` (newest 20% of train markets), `market_weights`, `fit_calibrator`. Settings, calibration and the
threshold come from the holdout; the test set is scored once.

Each model: `train.py` + `predictions/` (CSV outputs, gitignored — on S3 via `scripts/sync_predictions.py`) + `artifacts/` (gitignored binaries).
Prediction CSVs must include `market_id`, `snapshot_timestamp`, `outcome` and `dataset_version`.

To add a new model, use `/add-model <name>`.
