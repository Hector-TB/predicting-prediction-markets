"""
models/random_forest_trends/train.py
=====================================
Trains four Random Forest variants on the Google Trends-enriched dataset:
  - rf_price_only          — price_at_snapshot only
  - rf_full                — 21 engineered features (no trends)
  - rf_trends              — 26 features (full + 5 Google Trends columns)
  - rf_trends_calibrated   — rf_trends + isotonic calibration on held-out set

Mirrors models/random_forest/train.py but uses the with_trends dataset.
The extra Trends features are the primary comparison point (RQ2, RQ3).

Usage:
    python models/random_forest_trends/train.py
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT          = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = ROOT / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

sys.path.insert(0, str(ROOT))
from models.common.evaluation import (  # noqa: E402
    check_calibration,
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)

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
    "log_volume", "category_encoded",
]

FEATURES_TRENDS = FEATURES_FULL + [
    "trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data",
]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=50,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)


def load_and_split(data_path: Path):
    log.info("Loading %s ...", data_path.name)
    df = pd.read_parquet(data_path)
    df["category"] = df["category"].fillna("other")
    df = df.dropna(subset=[TARGET])
    log.info("  Total rows: %s | train: %s | test: %s",
             f"{len(df):,}",
             f"{(df['split']=='train').sum():,}",
             f"{(df['split']=='test').sum():,}")

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df["category_encoded"] = enc.fit_transform(df[["category"]]).astype(np.float32)

    train = df[df["split"] == "train"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)

    assert not (set(train["market_id"]) & set(test["market_id"])), "Leakage detected"

    rng     = np.random.default_rng(42)
    cal_ids = set(rng.choice(train["market_id"].unique(),
                             size=int(0.2 * train["market_id"].nunique()),
                             replace=False))
    train_fit = train[~train["market_id"].isin(cal_ids)].reset_index(drop=True)
    train_cal = train[ train["market_id"].isin(cal_ids)].reset_index(drop=True)

    log.info("  train_fit: %s rows | %s markets",
             f"{len(train_fit):,}", f"{train_fit['market_id'].nunique():,}")
    log.info("  train_cal: %s rows | %s markets",
             f"{len(train_cal):,}", f"{train_cal['market_id'].nunique():,}")
    return train_fit, train_cal, test


def make_sample_weights(y: np.ndarray, market_ids: np.ndarray) -> np.ndarray:
    """Class balance × inverse snapshot frequency."""
    class_w    = compute_sample_weight("balanced", y).astype(np.float32)
    counts     = pd.Series(market_ids).map(pd.Series(market_ids).value_counts()).values.astype(np.float32)
    snapshot_w = (1.0 / counts)
    snapshot_w = (snapshot_w / snapshot_w.mean()).astype(np.float32)
    combined   = class_w * snapshot_w
    return (combined / combined.mean()).astype(np.float32)


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           weights: np.ndarray, n_splits: int = 5) -> list[float]:
    cv     = StratifiedGroupKFold(n_splits=n_splits)
    aucs   = []
    log.info("\nRunning %d-fold CV on trends features (100 trees) ...", n_splits)
    cv_params = {**RF_PARAMS, "n_estimators": 100}
    for fold, (tr, val) in enumerate(cv.split(X, y, groups=groups), 1):
        clf = RandomForestClassifier(**cv_params)
        clf.fit(X[tr], y[tr], sample_weight=weights[tr])
        auc = roc_auc_score(y[val], clf.predict_proba(X[val])[:, 1])
        aucs.append(auc)
        log.info("  Fold %d: AUC = %.4f", fold, auc)
    log.info("  CV mean: %.4f ± %.4f", np.mean(aucs), np.std(aucs))
    return aucs


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_fit, train_cal, test = load_and_split(
        DATA_DIR / "polymarket_ml_dataset_with_trends_clean.parquet"
    )

    y_fit  = train_fit[TARGET].values.astype(np.int32)
    y_cal  = train_cal[TARGET].values.astype(np.int32)
    y_test = test[TARGET].values.astype(np.int32)

    X_fit_price   = train_fit[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_fit_full    = train_fit[FEATURES_FULL].values.astype(np.float32)
    X_fit_trends  = train_fit[FEATURES_TRENDS].values.astype(np.float32)
    X_cal_trends  = train_cal[FEATURES_TRENDS].values.astype(np.float32)
    X_test_price  = test[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_test_full   = test[FEATURES_FULL].values.astype(np.float32)
    X_test_trends = test[FEATURES_TRENDS].values.astype(np.float32)

    sample_weights = make_sample_weights(y_fit, train_fit["market_id"].values)

    run_cv(X_fit_trends, y_fit, train_fit["market_id"].values, sample_weights)

    log.info("\nTraining rf_price_only (300 trees) ...")
    rf_price = RandomForestClassifier(**RF_PARAMS)
    rf_price.fit(X_fit_price, y_fit, sample_weight=sample_weights)
    joblib.dump(rf_price, ARTIFACTS_DIR / "trends_rf_price_only.pkl")

    log.info("Training rf_full (300 trees) ...")
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X_fit_full, y_fit, sample_weight=sample_weights)
    joblib.dump(rf_full, ARTIFACTS_DIR / "trends_rf_full.pkl")

    log.info("Training rf_trends (300 trees) ...")
    rf_trends = RandomForestClassifier(**RF_PARAMS)
    rf_trends.fit(X_fit_trends, y_fit, sample_weight=sample_weights)
    joblib.dump(rf_trends, ARTIFACTS_DIR / "trends_rf_trends.pkl")

    log.info("Calibrating rf_trends with isotonic regression ...")
    p_cal_raw = rf_trends.predict_proba(X_cal_trends)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)
    joblib.dump({"rf": rf_trends, "iso": iso}, ARTIFACTS_DIR / "trends_rf_trends_calibrated.pkl")

    proba_price = rf_price.predict_proba(X_test_price)[:, 1]
    proba_full  = rf_full.predict_proba(X_test_full)[:, 1]
    proba_tr    = rf_trends.predict_proba(X_test_trends)[:, 1]
    proba_cal   = iso.transform(proba_tr)

    threshold, _ = find_optimal_threshold(y_test, proba_cal)

    evaluate(y_test, proba_price, label="RF Trends — price only", threshold=threshold)
    evaluate(y_test, proba_full,  label="RF Trends — full",       threshold=threshold)
    evaluate(y_test, proba_tr,    label="RF Trends — trends",     threshold=threshold)
    evaluate(y_test, proba_cal,   label="RF Trends — calibrated", threshold=threshold)
    evaluate_by_category(test, proba_cal, threshold=threshold)

    log.info("\nCalibration (before vs after):")
    log.info("  Before:")
    check_calibration(y_test, proba_tr)
    log.info("  After:")
    check_calibration(y_test, proba_cal)

    pred_df = test[["market_id", "snapshot_timestamp", TARGET, "split"]].copy()
    pred_df["proba_price_only"]        = proba_price.round(6)
    pred_df["proba_full"]              = proba_full.round(6)
    pred_df["proba_trends"]            = proba_tr.round(6)
    pred_df["proba_trends_calibrated"] = proba_cal.round(6)
    out = PREDICTIONS_DIR / "trends_test_predictions.csv"
    pred_df.to_csv(out, index=False)
    log.info("\nSaved %s rows → %s", f"{len(pred_df):,}", out)


if __name__ == "__main__":
    main()
