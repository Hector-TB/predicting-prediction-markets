"""
Shared evaluation utilities used across all model training scripts.

Import pattern (from any models/<name>/train.py):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.common.evaluation import evaluate, evaluate_by_category, find_optimal_threshold
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


TARGET = "outcome"


def find_optimal_threshold(y_true, y_prob):
    """Return (threshold, f1) that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(y_true, y_prob, label="", threshold=0.5):
    """Print a standard set of binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"  Threshold : {threshold:.3f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  PR-AUC    : {average_precision_score(y_true, y_prob):.4f}")
    print(f"  Log-loss  : {log_loss(y_true, y_prob):.4f}")
    print(f"  Brier     : {brier_score_loss(y_true, y_prob):.4f}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred):.4f}")
    print(f"{'─'*50}")


def evaluate_by_category(test_df, y_prob, threshold, target=TARGET):
    """Print per-category AUC, PR-AUC, F1, and YES rate."""
    print(f"\n{'─'*60}")
    print(f"  Per-category metrics (threshold={threshold:.3f})")
    print(f"  {'Category':<20} {'N':>7}  {'AUC':>6}  {'PR-AUC':>7}  {'F1':>6}  {'YES%':>6}")
    print(f"{'─'*60}")

    for cat in sorted(test_df["category"].unique()):
        mask = test_df["category"] == cat
        y_true_cat = test_df.loc[mask, target].values
        y_prob_cat = y_prob[mask.values]

        if len(np.unique(y_true_cat)) < 2 or len(y_true_cat) < 10:
            continue

        y_pred_cat = (y_prob_cat >= threshold).astype(int)
        auc = roc_auc_score(y_true_cat, y_prob_cat)
        pr  = average_precision_score(y_true_cat, y_prob_cat)
        f1  = f1_score(y_true_cat, y_pred_cat)
        yes = y_true_cat.mean()
        print(f"  {cat:<20} {mask.sum():>7,}  {auc:>6.4f}  {pr:>7.4f}  {f1:>6.4f}  {yes:>6.1%}")

    print(f"{'─'*60}")


def check_calibration(y_true, y_prob, n_bins=10):
    """Print a binned calibration table (predicted probability vs actual YES rate)."""
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    print(f"\n{'─'*50}")
    print(f"  Calibration (predicted → actual YES rate)")
    print(f"  {'Predicted':>10}  {'Actual':>10}  {'Diff':>8}")
    print(f"{'─'*50}")
    for pred, actual in zip(mean_predicted, fraction_of_positives):
        diff = actual - pred
        flag = "  <-- over-confident" if diff < -0.05 else (
               "  <-- under-confident" if diff > 0.05 else "")
        print(f"  {pred:>10.3f}  {actual:>10.3f}  {diff:>+8.3f}{flag}")
    print(f"{'─'*50}")
