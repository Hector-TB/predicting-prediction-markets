---
name: evaluate
description: Print current test metrics for all trained model variants vs the market baseline
disable-model-invocation: true
---

## Current model results

!`cd ${CLAUDE_PROJECT_DIR} && python3 scripts/print_metrics.py`

## Notes

- The baseline (market price as predictor) is computed live from `polymarket_ml_dataset_clean.parquet` — don't quote a fixed number
- Pre-ADR-014 results are not comparable with post-ADR-014 ones (see ADR-014 Consequences)
- To add a new model's predictions, add a row to `scripts/print_metrics.py` SOURCES list
