---
name: sync-and-train
description: Pull latest data from S3, refresh the data pipeline, then retrain all models
disable-model-invocation: true
---

Run these steps in order:

## 1. Pull data from S3

> **Skip this step if the local data is newer than S3** (e.g. right after a local pipeline
> rebuild that hasn't been pushed yet). `pull` downloads every file whose size differs from
> S3, so it would overwrite newer local files with the older S3 copies. Check with
> `python3 data/sync.py pull --dry-run` first when unsure.

```bash
python3 data/sync.py pull
```

If any files show "size mismatch", investigate before proceeding.

## 2. Refresh the data pipeline (optional — skip if data is already current)

```bash
python3 data_collection_pipeline/run_pipeline.py
python3 data/sync.py push
```

This fetches new resolved markets, extends Google Trends to today, and regenerates clean parquets.

## 3. Smoke test — verify data is healthy

```bash
python3 scripts/smoke_test.py
```

If it fails, stop and diagnose before training.

## 4. Train all models

```bash
python3 models/logistic_regression/train.py
python3 models/logistic_regression/train.py --trends
python3 models/gradient_boosting/train.py
python3 models/gradient_boosting/train.py --trends
python3 models/random_forest/train.py
python3 models/random_forest_trends/train.py
python3 models/svm/svm.py && python3 models/svm/svm_evaluate.py
```

## 5. Review results

Run `/evaluate` to see the updated metrics table, then re-score the paper's comparisons
(same test rows, market-level CIs, corrected trading ROI — ADR-015):

```bash
python3 analysis/rescore_paper.py --out docs/paper/rescore_rebuilt_data.md
```

## 6. Push updated predictions to S3

```bash
python3 data/sync.py push
```

## Notes

- Training scripts read local parquet directly — the pull in step 1 is required on a new machine
- SVM predictions use `target_percentile`, not standard probabilities — excluded from `/evaluate`
- RF models write to `models/random_forest/predictions/test_predictions.csv` and
  `models/random_forest_trends/predictions/trends_test_predictions.csv`
