# ADR-024: One Training Protocol for Every Model — Time-Ordered Holdout, Market Weights, Calibration

**Date:** 2026-09-29  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  
**Supersedes:** ADR-021 decisions 2–3 (how thresholds and LR settings are chosen); the rest of ADR-021 stands  
**Code location:** `models/common/training.py`; `models/{logistic_regression,gradient_boosting,random_forest,random_forest_trends}/train.py`

---

## Context

A review before the first v3 run found the four trained models each did tuning differently:

| | LR | XGBoost | RF / RF + Trends |
|---|---|---|---|
| Settings chosen on | 5-fold CV, grouped by market | a **random** 20% of train markets | the same random 20% |
| Calibration | **none** | isotonic on the 20% | isotonic on the 20% |
| Threshold from | out-of-fold predictions | the 20% | the 20% |
| One-weight-per-market | yes | **no** | yes |
| Final model trained on | all of train | 80% | 80% |

Three consequences:

1. **LR's probabilities were not probabilities.** `class_weight='balanced'` pushes every score towards YES. Without calibration, LR's log-loss and its trading-simulation edges (`p_model − price`) were biased by construction, not by what it learned.
2. **XGBoost learned mostly from long markets.** Train markets have 1 to 1,065 snapshots each (median 43). Every snapshot counted equally, so one long market outweighed hundreds of short ones. v3 adds more late rows to long markets. The model comparison partly measured this weighting difference.
3. **The random holdout didn't look like the test set.** The train/test split is by time (ADR-021) and the YES rate has risen over time: 17% of markets resolving in 2025 Q4, 23%, 28%, then 33% in 2026 Q3, in every category. Test snapshots are 38.4% YES against 25.4% in train. A random 20% of train markets matches the train average, so calibration and thresholds were fitted to older conditions than the ones they are applied to.

## Decision

Every model (LR, XGBoost, RF, RF + Trends; ± Trends) follows the same protocol, from `models/common/training.py`:

1. **Holdout by time.** Sort train markets by resolution time (`closedTime`, else scheduled `end_date`, the same rule as `recompute_split.py`). The oldest 80% are **fit**, the newest 20% are **holdout**. Holdout rows dated before the last fit market resolved are dropped. This is the ADR-021 split rule applied again inside train, so no label the model learned from postdates a holdout snapshot.
2. **Weights.** Each row gets 1 / (its market's snapshot count), scaled to mean 1, so every market counts once in total. This is on top of the library's class balancing (`class_weight='balanced'` or `scale_pos_weight`, ADR-007).
3. **Settings are chosen by AUC on the holdout.** XGBoost also early-stops on the holdout AUC. AUC is used because calibration comes afterwards and doesn't change the ranking.
4. **The final model is fitted on the fit markets** with the chosen settings. It is not refitted on all of train: the holdout must stay unseen by the model so it can be used for calibration.
5. **Isotonic calibration and the F1 threshold both come from the holdout.**
6. **The test set is scored once**, after everything above is fixed.

Prediction files keep their columns. LR now writes `pred_prob_raw` and a calibrated `pred_prob`, like XGBoost.

Also removed: RF's 5-fold CV (it only printed a number and chose nothing), and `evaluate_by_volume_quintile` (it grouped markets by final lifetime volume, the look-ahead value ADR-018 dropped).

## Rationale

- **One protocol** means differences between models come from the models, which is what the paper compares.
- **Single holdout, not k-fold:** XGBoost's grid alone is 27 fits of up to 1,000 trees on ~1.7M rows and 2 CPU cores. 5-fold would be 135 fits. A single holdout also gives the unseen predictions that calibration needs, without extra fits.
- **By time, not random:** it follows ADR-021's rule (train on the past, predict the future). It is also the newest data available before the cutoff T, so calibration reflects conditions closest to the test period.

## Alternatives considered

- **Time-ordered k-fold for all models (walk-forward).** More stable estimates, about 5× the compute. Revisit with the one-command refresh (ADR-019), as ADR-021 already notes.
- **Refit on all of train and calibrate with cross-validated predictions** (`CalibratedClassifierCV`). Uses 20% more data, but needs k fits and a different early-stopping setup for XGBoost. Rejected for now.
- **Correct for the YES-rate rise by reweighting classes to the test rate.** That would use the test set's label mix, which is a leak. Rejected. The time-ordered holdout is the leak-free way to track recent conditions.
- **Separate holdouts for tuning and for calibration.** This avoids reusing one holdout for four decisions, but splits an already small holdout. The reuse cannot touch the test set; at worst calibration is slightly optimistic. Accepted.

## Assumptions

1. The recent past is a better guide to the test period than the average past. The steady YES-rate rise supports this.
2. The holdout is large enough for isotonic calibration. On v3 it has 84,532 rows and 3,313 markets (fit: 1,717,518 rows, 18,496 markets). Holdout cutoff: 2026-05-29 05:56 UTC, so the holdout covers roughly the last month before T.

## Consequences

- The holdout's YES rate on v3 is 31.4% (fit: 24.7%, test: 38.4%).
- 236,845 train rows (holdout rows dated before the holdout cutoff) are used for neither fitting nor calibration, just as `test_pre_cutoff` rows are unused.
- **Results are not comparable** with runs before this ADR.
- LR's log-loss and trading results change most, because it is now calibrated.
- SVM (`models/svm/`) is not covered: it uses its own subsample (ADR-006) and is not part of the paper re-score.
- `analysis/rescore_paper.py` now skips prediction files whose `dataset_version` doesn't match the local data. It also reports AUC by time left before the scheduled close, to show whether v3's late rows help (ADR-022).

## Related ADRs

- ADR-007: Class imbalance (market weights added on top, see its amendment)
- ADR-018: Volume look-ahead (volume-quintile report removed)
- ADR-021: Leak-free evaluation (split rule reused for the holdout; decisions 2–3 superseded)
- ADR-022: No pre-close cutoff (the time-left report)
