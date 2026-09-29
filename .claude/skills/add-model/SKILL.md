---
name: add-model
description: Scaffold a new model directory following project conventions
disable-model-invocation: true
argument-hint: "[model-name]"
---

Scaffold a new model at `models/$ARGUMENTS/` following the project pattern.

## Steps

1. Create the directory structure:
```
models/$ARGUMENTS/
├── train.py
├── predictions/       (empty, gitignored predictions go here)
└── artifacts/         (empty, gitignored model artifacts go here)
```

2. `train.py` must follow this template:
   - `ROOT = Path(__file__).resolve().parent.parent.parent`
   - `DATA_DIR = ROOT / "data"`
   - Import `evaluate`, `check_calibration`, `find_optimal_threshold` from `models.common.evaluation`
   - Follow the shared training protocol (ADR-024): `load_dataset`, `split_holdout`, `market_weights`,
     `fit_calibrator` from `models.common.training`. Choose settings by AUC on the holdout, calibrate and
     pick the threshold on the holdout, score the test set once
   - Use `class_weight='balanced'` or equivalent, plus `sample_weight=market_weights(...)`
   - Write predictions to `predictions/predictions.csv` with columns: `market_id`, `snapshot_timestamp`, `outcome`, `dataset_version` (from `models.common.evaluation.dataset_version()`), and the probability column(s)
   - Use logging, not bare print()

3. After creating `train.py`, add the new model to:
   - `scripts/print_metrics.py` SOURCES list (with the prob column name)
   - `analysis/rescore_paper.py` MODELS list
   - `db/load_parquet.py` `load_model_runs()` sources list (with dataset_tag)
   - `README.md` Results table

4. Run the smoke test to confirm the environment works before training:
```bash
python3 scripts/smoke_test.py
```

## What NOT to do

- Do not define `evaluate()`, `check_calibration()`, or `find_optimal_threshold()` in the new file
- Do not import `roc_auc_score`, `log_loss`, `f1_score` directly — use `models.common.evaluation`
- Do not hardcode data paths
