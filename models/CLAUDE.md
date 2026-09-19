# Models

Shared evaluation utilities live in `models/common/evaluation.py`.
Import from there — do not redefine `evaluate`, `check_calibration`, or `find_optimal_threshold`.

Each model: `train.py` + `predictions/` (CSV outputs) + `artifacts/` (gitignored binaries).

To add a new model, use `/add-model <name>`.
