"""
Shared evaluation utilities used across all model training scripts.

Import pattern (from any models/<name>/train.py):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models.common.evaluation import evaluate, evaluate_by_category, find_optimal_threshold
"""

import numpy as np
import pandas as pd
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


def bootstrap_auc_diff(y_true, y_prob_base, y_prob_new, n_boot: int = 1000, seed: int = 42):
    """
    Bootstrap 95% CI on (AUC_new − AUC_base).

    Reports the point estimate, CI, and a one-sided p-value for H0: diff <= 0.
    Use this to test whether a Trends-enriched model genuinely beats its baseline.
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        yt   = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(
            roc_auc_score(yt, y_prob_new[idx]) - roc_auc_score(yt, y_prob_base[idx])
        )
    diffs = np.array(diffs)
    point  = roc_auc_score(y_true, y_prob_new) - roc_auc_score(y_true, y_prob_base)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_val  = (diffs <= 0).mean()

    print(f"\n{'─'*55}")
    print(f"  Bootstrap AUC difference (new − base)  [n_boot={n_boot}]")
    print(f"  Point estimate : {point:+.4f}")
    print(f"  95% CI         : [{lo:+.4f},  {hi:+.4f}]")
    if lo > 0:
        verdict = "significant (CI excludes 0)"
    elif hi < 0:
        verdict = "significant NEGATIVE (CI excludes 0)"
    else:
        verdict = "not significant (CI includes 0)"
    print(f"  p (diff <= 0)  : {p_val:.3f}  →  {verdict}")
    print(f"{'─'*55}")
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "p_value": p_val}


def evaluate_by_volume_quintile(test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float,
                                target: str = TARGET):
    """
    Per-volume-quintile AUC, PR-AUC, and F1 on the test set.

    Helps identify whether the model's edge concentrates in thin markets
    (low liquidity / slow price discovery) vs high-volume ones.
    """
    df = test_df.copy().reset_index(drop=True)
    df["_prob"] = y_prob

    # Aggregate to market level (mean prediction, single outcome per market)
    mkt = (
        df.groupby("market_id")
        .agg(
            total_volume=("total_volume", "first"),
            outcome=(target, "first"),
            prob=("_prob", "mean"),
        )
        .reset_index()
    )
    mkt["quintile"] = pd.qcut(mkt["total_volume"], q=5,
                               labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])

    print(f"\n{'─'*72}")
    print(f"  Per-volume-quintile metrics (market-level, threshold={threshold:.3f})")
    print(f"  {'Quintile':<14} {'N':>6}  {'Volume range':>22}  {'AUC':>6}  {'PR-AUC':>7}  {'YES%':>6}")
    print(f"{'─'*72}")

    for q in mkt["quintile"].cat.categories:
        grp  = mkt[mkt["quintile"] == q]
        yt   = grp["outcome"].values
        yp   = grp["prob"].values
        vmin = grp["total_volume"].min()
        vmax = grp["total_volume"].max()
        if len(np.unique(yt)) < 2:
            print(f"  {str(q):<14} {len(grp):>6}  (skip — single class)")
            continue
        auc = roc_auc_score(yt, yp)
        pr  = average_precision_score(yt, yp)
        yes = yt.mean()
        print(f"  {str(q):<14} {len(grp):>6}  ${vmin:>9,.0f}–${vmax:>9,.0f}  {auc:>6.4f}  {pr:>7.4f}  {yes:>6.1%}")

    print(f"{'─'*72}")


def analyze_market_disagreements(test_df: pd.DataFrame, y_prob: np.ndarray,
                                 threshold: float, target: str = TARGET, top_n: int = 5):
    """
    Find markets where the model disagrees with market price and is correct.

    'Market price' here is the mean price_at_snapshot across that market's test
    snapshots — a proxy for the crowd's running consensus.  Markets where the
    model wins are candidates for live-scoring alpha.
    """
    df = test_df.copy().reset_index(drop=True)
    df["_prob"] = y_prob

    mkt = (
        df.groupby("market_id")
        .agg(
            outcome=(target, "first"),
            model_prob=("_prob", "mean"),
            market_prob=("price_at_snapshot", "mean"),
            category=("category", "first"),
            total_volume=("total_volume", "first"),
            question=("question", "first") if "question" in df.columns else ("market_id", "first"),
        )
        .reset_index()
    )

    mkt["model_pred"]  = (mkt["model_prob"]  >= threshold).astype(int)
    mkt["market_pred"] = (mkt["market_prob"] >= 0.5).astype(int)
    mkt["model_right"]  = mkt["model_pred"]  == mkt["outcome"]
    mkt["market_right"] = mkt["market_pred"] == mkt["outcome"]

    model_only_right = mkt[mkt["model_right"] & ~mkt["market_right"]]
    model_only_wrong = mkt[~mkt["model_right"] & mkt["market_right"]]

    n_total = len(mkt)
    print(f"\n{'─'*60}")
    print(f"  Market-level disagreement analysis  ({n_total:,} markets)")
    print(f"  Model right, market wrong : {len(model_only_right):>5,}  ({len(model_only_right)/n_total:.1%})")
    print(f"  Market right, model wrong : {len(model_only_wrong):>5,}  ({len(model_only_wrong)/n_total:.1%})")

    for label, subset in [("Model right / market wrong", model_only_right),
                           ("Market right / model wrong", model_only_wrong)]:
        if subset.empty:
            continue
        print(f"\n  {label} — top {top_n} by volume:")
        top = subset.nlargest(top_n, "total_volume")
        for _, row in top.iterrows():
            q = str(row.get("question", row["market_id"]))[:60]
            print(f"    [{row['category']:<12}]  vol=${row['total_volume']:>10,.0f}  "
                  f"model={row['model_prob']:.2f}  market={row['market_prob']:.2f}  "
                  f"outcome={'YES' if row['outcome'] else 'NO '}  "
                  f"  {q}")
    print(f"{'─'*60}")
