"""
Print a metrics table for all trained model variants vs the market baseline.
Used by the /evaluate skill and for ad-hoc comparison.

    python scripts/print_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SOURCES = [
    # (label,            csv path,                                                   prob col)
    ("Market baseline",  None,                                                        "price_at_snapshot"),
    ("LR base",          MODELS / "logistic_regression/predictions/predictions.csv",  "pred_prob_base"),
    ("LR + trends",      MODELS / "logistic_regression/predictions/predictions.csv",  "pred_prob_trends"),
    ("XGBoost base",     MODELS / "gradient_boosting/predictions/predictions.csv",    "pred_prob_base"),
    ("XGBoost + trends", MODELS / "gradient_boosting/predictions/predictions.csv",    "pred_prob_trends"),
    ("RF price-only",    MODELS / "random_forest/predictions/test_predictions.csv",   "proba_price_only"),
    ("RF full",          MODELS / "random_forest/predictions/test_predictions.csv",   "proba_full"),
    ("RF calibrated",    MODELS / "random_forest/predictions/test_predictions.csv",   "proba_full_calibrated"),
    ("RFT full",         MODELS / "random_forest_trends/predictions/trends_test_predictions.csv", "proba_full"),
    ("RFT calibrated",   MODELS / "random_forest_trends/predictions/trends_test_predictions.csv", "proba_trends_calibrated"),
]

BASELINE_PATH = ROOT / "data" / "polymarket_ml_dataset_clean.parquet"


def compute(y, p):
    mask = ~np.isnan(p)
    y, p = y[mask], p[mask]
    return {
        "auc":  roc_auc_score(y, p),
        "ll":   log_loss(y, p),
        "pr":   average_precision_score(y, p),
        "n":    len(y),
    }


def main():
    cache: dict[Path, pd.DataFrame] = {}

    # Market baseline — load from parquet if CSV not available
    baseline = None
    if BASELINE_PATH.exists():
        df_base = pd.read_parquet(BASELINE_PATH, columns=["price_at_snapshot", "outcome", "split"])
        df_base = df_base[df_base["split"] == "test"]
        baseline = compute(df_base["outcome"].values, df_base["price_at_snapshot"].values)

    print(f"\n  {'Model':<22} {'AUC-ROC':>8} {'Log-loss':>10} {'PR-AUC':>8} {'N':>10}")
    print("  " + "─" * 64)

    for label, csv_path, prob_col in SOURCES:
        if csv_path is None:
            if baseline:
                m = baseline
                print(f"  {'Market baseline':<22} {m['auc']:>8.4f} {m['ll']:>10.4f} {m['pr']:>8.4f} {m['n']:>10,}  ← beat this")
            else:
                print(f"  {'Market baseline':<22} {'0.9640':>8} {'0.1710':>10} {'':>8} {'':>10}  ← beat this")
            print("  " + "─" * 64)
            continue

        if not csv_path.exists():
            print(f"  {label:<22}  (CSV not found)")
            continue

        if csv_path not in cache:
            cache[csv_path] = pd.read_csv(csv_path)
        df = cache[csv_path]

        if prob_col not in df.columns:
            print(f"  {label:<22}  (column '{prob_col}' not found)")
            continue

        m = compute(df["outcome"].values, df[prob_col].values)
        print(f"  {label:<22} {m['auc']:>8.4f} {m['ll']:>10.4f} {m['pr']:>8.4f} {m['n']:>10,}")

    print()


if __name__ == "__main__":
    main()
