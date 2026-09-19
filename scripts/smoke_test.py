"""
Quick sanity check — not a regression test.
Loads a sample of rows, fits a minimal logistic regression, verifies AUC > floor.
Run after pipeline changes or environment updates to confirm nothing is broken.

    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "polymarket_ml_dataset_clean.parquet"
AUC_FLOOR = 0.80

FEATURES = [
    "price_at_snapshot", "days_before_close", "pct_lifetime_elapsed",
    "price_mean_7d", "price_volatility_7d", "price_change_7d",
    "price_mean_14d", "price_volatility_14d", "price_change_14d",
    "log_volume",
]
TARGET = "outcome"


def main():
    if not DATA.exists():
        print(f"SKIP: {DATA.name} not found — run: python data/sync.py pull")
        sys.exit(0)

    print(f"Loading sample from {DATA.name} ...")
    df = pd.read_parquet(DATA, columns=FEATURES + [TARGET, "split"])

    train = df[df["split"] == "train"].dropna(subset=FEATURES).sample(n=2000, random_state=42)
    test  = df[df["split"] == "test"].dropna(subset=FEATURES).sample(n=500,  random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test  = scaler.transform(test[FEATURES])

    model = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
    model.fit(X_train, train[TARGET])

    auc    = roc_auc_score(test[TARGET], model.predict_proba(X_test)[:, 1])
    status = "PASS" if auc >= AUC_FLOOR else "FAIL"

    print(f"AUC-ROC: {auc:.4f}  (floor: {AUC_FLOOR})  [{status}]")

    if auc < AUC_FLOOR:
        print(f"\nFAIL: AUC {auc:.4f} is below the floor of {AUC_FLOOR}.")
        print("This suggests the data or feature pipeline is broken.")
        sys.exit(1)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
