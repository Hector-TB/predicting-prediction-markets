---
name: evaluate
description: Print current test metrics for all trained model variants vs the market baseline
disable-model-invocation: true
---

## Current model results

!`cd ${CLAUDE_PROJECT_DIR} && python3 scripts/print_metrics.py`

## Notes

- Baseline AUC-ROC = 0.964, log-loss = 0.171 (market price as predictor)
- No model currently beats the baseline — see ADR-011 and the Results section in README
- To add a new model's predictions, add a row to `scripts/print_metrics.py` SOURCES list
