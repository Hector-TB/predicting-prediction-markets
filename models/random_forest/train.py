"""
models/random_forest/train.py
==============================
Trains three Random Forest variants on base features (no Google Trends):
  - rf_price_only        — price_at_snapshot only
  - rf_full              — 20 engineered features
  - rf_full_calibrated   — rf_full + isotonic calibration on the holdout

Shared training protocol (ADR-024): max_depth × min_samples_leaf chosen by AUC
on the time-ordered holdout (100 trees per candidate, 300 for the winner);
calibration and threshold from the same holdout; test scored once.

Usage:
    python models/random_forest/train.py
"""

import itertools
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

ROOT          = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

sys.path.insert(0, str(ROOT))
from models.common.evaluation import (  # noqa: E402
    dataset_version,
    analyze_market_disagreements,
    check_calibration,
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)
from models.common.training import fit_calibrator, load_dataset, market_weights, split_holdout  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TARGET = "outcome"

FEATURES_PRICE_ONLY = ["price_at_snapshot"]

FEATURES_FULL = [
    "price_at_snapshot", "price_deviation_from_half",
    "price_mean_7d",  "price_volatility_7d",  "price_min_7d",  "price_max_7d",
    "price_change_7d",  "price_range_7d",  "price_trend_7d",
    "price_mean_14d", "price_volatility_14d", "price_min_14d", "price_max_14d",
    "price_change_14d", "price_range_14d", "price_trend_14d",
    "pct_lifetime_elapsed", "days_before_close", "duration_days",
    "category_encoded",  # log_volume dropped: final lifetime volume = look-ahead (ADR-018)
]

GRID = {
    "max_depth":        [8, 12, 16, 20],
    "min_samples_leaf": [20, 50, 100],
}

RF_BASE = dict(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


def load_and_split(trends: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(trends=trends)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["category_encoded"] = enc.fit_transform(df[["category"]]).astype(np.float32)
    log.info("  Categories: %s", enc.categories_[0].tolist())

    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"].reset_index(drop=True)
    assert not (set(train["market_id"]) & set(test["market_id"])), "Leakage detected"

    fit, holdout = split_holdout(train)
    return fit, holdout, test


def run_grid_search(
    X_fit: np.ndarray, y_fit: np.ndarray, w_fit: np.ndarray,
    X_hold: np.ndarray, y_hold: np.ndarray,
) -> dict:
    """Grid-search max_depth × min_samples_leaf by AUC on the holdout (100 trees)."""
    combos = list(itertools.product(GRID["max_depth"], GRID["min_samples_leaf"]))
    log.info("\nGrid search: %d combinations (100 trees each) ...", len(combos))

    best_auc, best_params = -1.0, {}
    for i, (depth, leaf) in enumerate(combos, 1):
        clf = RandomForestClassifier(**{**RF_BASE, "n_estimators": 100}, max_depth=depth, min_samples_leaf=leaf)
        clf.fit(X_fit, y_fit, sample_weight=w_fit)
        auc = roc_auc_score(y_hold, clf.predict_proba(X_hold)[:, 1])
        log.info("  [%2d/%d]  depth=%-3d  leaf=%-4d  AUC=%.4f%s",
                 i, len(combos), depth, leaf, auc, " *" if auc > best_auc else "")
        if auc > best_auc:
            best_auc, best_params = auc, {"max_depth": depth, "min_samples_leaf": leaf}

    log.info("  Best: %s  →  AUC=%.4f", best_params, best_auc)
    return best_params


def log_shap(clf: RandomForestClassifier, X: np.ndarray, features: list[str], top_n: int = 10) -> None:
    log.info("\nSHAP feature attribution (TreeExplainer, sample=5000) ...")
    try:
        import shap
        idx = np.random.default_rng(42).choice(len(X), size=min(5000, len(X)), replace=False)
        sv = shap.TreeExplainer(clf).shap_values(X[idx])
        # Older shap returns [class0, class1]; newer returns (rows, features, classes)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = sv[..., 1] if sv.ndim == 3 else sv
        ranked = sorted(zip(features, np.abs(sv).mean(axis=0)), key=lambda x: x[1], reverse=True)
        log.info("  Top features by mean |SHAP|:")
        for feat, score in ranked[:top_n]:
            log.info("    %+.4f  %-30s  %s", score, feat, "█" * max(1, int(score * 300)))
    except Exception as e:
        log.warning("  SHAP skipped: %s", e)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    fit, holdout, test = load_and_split()

    y_fit  = fit[TARGET].values.astype(np.int32)
    y_hold = holdout[TARGET].values.astype(np.int32)
    y_test = test[TARGET].values.astype(np.int32)

    X_fit_price  = fit[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_fit_full   = fit[FEATURES_FULL].values.astype(np.float32)
    X_hold_full  = holdout[FEATURES_FULL].values.astype(np.float32)
    X_test_price = test[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_test_full  = test[FEATURES_FULL].values.astype(np.float32)

    w_fit = market_weights(fit["market_id"])

    best_params = run_grid_search(X_fit_full, y_fit, w_fit, X_hold_full, y_hold)
    rf_params   = {**RF_BASE, **best_params}

    log.info("\nTraining rf_price_only (300 trees, default depth=16 leaf=50) ...")
    rf_price = RandomForestClassifier(**RF_BASE, max_depth=16, min_samples_leaf=50)
    rf_price.fit(X_fit_price, y_fit, sample_weight=w_fit)
    joblib.dump(rf_price, ARTIFACTS_DIR / "rf_price_only.pkl")
    log.info("  Saved rf_price_only.pkl")

    log.info("Training rf_full (300 trees, best params) ...")
    rf_full = RandomForestClassifier(**rf_params)
    rf_full.fit(X_fit_full, y_fit, sample_weight=w_fit)
    joblib.dump(rf_full, ARTIFACTS_DIR / "rf_full.pkl")
    log.info("  Saved rf_full.pkl (params: %s)", best_params)

    log.info("Calibrating rf_full with isotonic regression on the holdout ...")
    p_hold_raw = rf_full.predict_proba(X_hold_full)[:, 1]
    iso = fit_calibrator(p_hold_raw, y_hold)

    # Threshold chosen on the holdout — never on the test set (ADR-021)
    threshold, _ = find_optimal_threshold(y_hold, iso.transform(p_hold_raw))
    joblib.dump({"rf": rf_full, "iso": iso, "threshold": threshold}, ARTIFACTS_DIR / "rf_full_calibrated.pkl")
    log.info("  Saved rf_full_calibrated.pkl")

    proba_price = rf_price.predict_proba(X_test_price)[:, 1]
    proba_full  = rf_full.predict_proba(X_test_full)[:, 1]
    proba_cal   = iso.transform(proba_full)

    evaluate(y_test, proba_price, label="RF — price only",  threshold=threshold)
    evaluate(y_test, proba_full,  label="RF — full",        threshold=threshold)
    evaluate(y_test, proba_cal,   label="RF — calibrated",  threshold=threshold)
    evaluate_by_category(test, proba_cal, threshold=threshold)
    analyze_market_disagreements(test, proba_cal, threshold=threshold)

    log.info("\nCalibration (before vs after):")
    log.info("  Before:")
    check_calibration(y_test, proba_full)
    log.info("  After:")
    check_calibration(y_test, proba_cal)

    log_shap(rf_full, X_test_full, FEATURES_FULL)

    pred_df = test[["market_id", "snapshot_timestamp", TARGET, "split"]].copy()
    pred_df["dataset_version"]       = dataset_version()
    pred_df["proba_price_only"]      = proba_price.round(6)
    pred_df["proba_full"]            = proba_full.round(6)
    pred_df["proba_full_calibrated"] = proba_cal.round(6)
    out = PREDICTIONS_DIR / "test_predictions.csv"
    pred_df.to_csv(out, index=False)
    log.info("\nSaved %s rows → %s", f"{len(pred_df):,}", out)


if __name__ == "__main__":
    main()
