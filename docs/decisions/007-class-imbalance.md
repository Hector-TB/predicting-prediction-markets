# ADR-007: Class Imbalance Handling

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `models/*/train.py` and `models/svm/svm.py`

---

## Context

At the market level, **80.3% of markets resolve NO** and only 19.7% resolve YES. In the snapshot dataset the imbalance is slightly less severe (77.9% NO / 22.1% YES) because YES markets tend to be longer-lived.

A model that always predicts NO achieves ~78% accuracy — making raw accuracy meaningless as a metric.

## Decision

All models apply cost-sensitive learning to handle imbalance:

- **Logistic Regression / SVM:** `class_weight='balanced'` — sklearn automatically sets class weights inversely proportional to class frequencies
- **XGBoost:** `scale_pos_weight = (count of negatives) / (count of positives)` — equivalent mechanism native to XGBoost
- **Primary metrics:** AUC-ROC and log-loss (insensitive to threshold choice); **not raw accuracy**
- **Threshold selection:** optimal threshold found by maximising F1 on the precision-recall curve, not defaulting to 0.5

## Rationale

**Why cost-sensitive over oversampling/undersampling:** cost-sensitive learning modifies the loss function without changing the data; SMOTE and undersampling create or destroy data points, adding complexity. For this problem size cost-sensitive is simpler and equally effective.

**Why not report accuracy:** A model that always predicts NO hits 78% accuracy. Reporting accuracy as a primary metric would mislead.

**Why AUC-ROC:** Threshold-independent, intuitive (probability that model ranks a YES higher than a NO), and the standard metric for the baseline comparison (market price AUC = 0.964).

## Alternatives considered

- **SMOTE oversampling** — adds synthetic minority samples; effective but adds a preprocessing step and complicates the pipeline; not needed given the moderate imbalance ratio (~4:1)
- **Random undersampling** — discards majority-class data; wasteful given the large dataset
- **No correction** — would produce models biased toward predicting NO; rejected

## Assumptions

1. The 4:1 imbalance is stable across splits (empirically confirmed: train 22.4% YES, test 21.0% YES)
2. Cost-sensitive learning provides sufficient correction at this imbalance ratio

## Consequences

- Models may appear to have lower raw accuracy than a naive NO-classifier, but higher AUC and F1
- Threshold at 0.5 is no longer the right operating point; optimal threshold is found empirically per model
- All evaluation reporting must include AUC-ROC and log-loss; raw accuracy may be included as a secondary metric only

## Related ADRs

- ADR-004: Outcome threshold (source of the imbalance — 80% of markets resolve NO)
