---
name: sync-and-train
description: Pull latest data from S3 then run training for all models
disable-model-invocation: true
---

Run these steps in order:

## 1. Pull data from S3

```bash
python3 data/sync.py pull
```

If any files show "size mismatch", investigate before proceeding.

## 2. Smoke test — verify data is healthy

```bash
python3 scripts/smoke_test.py
```

If it fails, stop and diagnose the data pipeline before training.

## 3. Train models

```bash
python3 models/logistic_regression/train.py
python3 models/gradient_boosting/train.py
python3 models/svm/svm.py && python3 models/svm/svm_evaluate.py
```

## 4. Review results

Run `/evaluate` to see the updated metrics table.

## 5. Push updated predictions to S3 (if predictions CSVs changed)

```bash
python3 data/sync.py push
```

## Notes

- Training scripts read local parquet directly — the pull in step 1 is required on a new machine
- SVM predictions use `target_percentile`, not standard probabilities — excluded from `/evaluate`
- random_forest and lightgbm have no `train.py` yet (notebook only)
