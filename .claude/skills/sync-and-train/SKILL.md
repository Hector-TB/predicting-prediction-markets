---
name: sync-and-train
description: Pull latest data from S3, refresh the data pipeline, then retrain all models
disable-model-invocation: true
---

Run these steps in order:

## 1. Pull data from S3

> **Skip this step right after a local pipeline rebuild** — publish it first (step 2).
> `pull` refuses to overwrite local files that differ from the S3 version unless `--force`
> is given. Check what's local with `python3 data/sync.py status`.

```bash
python3 data/sync.py pull
```

If any files show "size mismatch", investigate before proceeding.

## 2. Refresh the data pipeline (optional — skip if data is already current)

```bash
python3 data_collection_pipeline/run_pipeline.py
python3 scripts/coverage_check.py                 # no qualifying markets missed (ADR-020)
python3 scripts/check_snapshots.py                # full quality check vs the latest version
python3 scripts/stage_backup.py --name <date>-<next version>   # optional insurance before publishing
python3 data/sync.py publish <next version> --parent <current> --notes "…" --dry-run
python3 data/sync.py publish <next version> --parent <current> --notes "…"
```

Versions are immutable (ADR-016); `python3 data/sync.py list` shows what exists. The pipeline fetches new resolved markets, refreshes Google Trends, and regenerates the clean parquets. It stops if the market metadata doesn't match Gamma (ADR-023). Don't publish unless both checks pass.

> Until ADR-020's incremental fetch is built, a plain incremental `fetch_markets.py` can miss long-running markets. Run `coverage_check.py` after any fetch.

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
python3 analysis/rescore_paper.py --out docs/paper/rescore_<version>_data.md
```

## 6. Record results

Prediction CSVs are tracked in git — commit them. Model runs in the DB are tagged with the
dataset version (`clean_<version>`), so re-run `python3 db/load_parquet.py` after training.

## Notes

- Training scripts read local parquet directly — the pull in step 1 is required on a new machine
- SVM predictions use `target_percentile`, not standard probabilities — excluded from `/evaluate`
- RF models write to `models/random_forest/predictions/test_predictions.csv` and
  `models/random_forest_trends/predictions/trends_test_predictions.csv`
