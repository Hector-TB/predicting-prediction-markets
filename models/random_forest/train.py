"""
models/random_forest/train.py
==============================
Trains three Random Forest variants on base features (no Google Trends):
  - rf_price_only        — price_at_snapshot only
  - rf_full              — 21 engineered features
  - rf_full_calibrated   — rf_full + isotonic calibration on held-out set

Grid-searches max_depth × min_samples_leaf on the validation set before
the final fit (100 trees per candidate, 300 for the winner).

Sample weights combine class balance with inverse snapshot frequency so
high-frequency markets don't dominate (see ADR-007).

Usage:
    python models/random_forest/train.py
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

ROOT          = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = ROOT / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

sys.path.insert(0, str(ROOT))
from models.common.evaluation import (  # noqa: E402
    analyze_market_disagreements,
    check_calibration,
    evaluate,
    evaluate_by_category,
    evaluate_by_volume_quintile,
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

GRID = {
    "max_depth":        [8, 12, 16, 20],
    "min_samples_leaf": [20, 50, 100],
}

RF_BASE = dict(
    n_estimators=300,
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
    log.info("  Categories: %s", enc.categories_[0].tolist())

    train = df[df["split"] == "train"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)

    assert not (set(train["market_id"]) & set(test["market_id"])), "Leakage detected"

    # Hold out 20% of train markets for isotonic calibration
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
    """Class balance × inverse snapshot frequency — prevents high-volume markets dominating."""
    from sklearn.utils.class_weight import compute_sample_weight
    class_w    = compute_sample_weight("balanced", y).astype(np.float32)
    counts     = pd.Series(market_ids).map(pd.Series(market_ids).value_counts()).values.astype(np.float32)
    snapshot_w = (1.0 / counts)
    snapshot_w = (snapshot_w / snapshot_w.mean()).astype(np.float32)
    combined   = class_w * snapshot_w
    return (combined / combined.mean()).astype(np.float32)


def run_grid_search(
    X_fit: np.ndarray, y_fit: np.ndarray, weights_fit: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> dict:
    """Grid-search max_depth × min_samples_leaf using the held-out val set (100 trees)."""
    import itertools
    combos = list(itertools.product(GRID["max_depth"], GRID["min_samples_leaf"]))
    log.info("\nGrid search: %d combinations (100 trees each) ...", len(combos))

    best_auc    = -1.0
    best_params: dict = {}

    for i, (depth, leaf) in enumerate(combos, 1):
        clf = RandomForestClassifier(**RF_BASE, n_estimators=100, max_depth=depth, min_samples_leaf=leaf)
        clf.fit(X_fit, y_fit, sample_weight=weights_fit)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        marker = " *" if auc > best_auc else ""
        log.info("  [%2d/%d]  depth=%-3d  leaf=%-4d  AUC=%.4f%s",
                 i, len(combos), depth, leaf, auc, marker)
        if auc > best_auc:
            best_auc    = auc
            best_params = {"max_depth": depth, "min_samples_leaf": leaf}

    log.info("  Best: %s  →  AUC=%.4f", best_params, best_auc)
    return best_params


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           weights: np.ndarray, rf_params: dict, n_splits: int = 5) -> list[float]:
    cv   = StratifiedGroupKFold(n_splits=n_splits)
    aucs = []
    log.info("\nRunning %d-fold CV (100 trees) ...", n_splits)
    cv_params = {**RF_BASE, **rf_params, "n_estimators": 100}
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
        DATA_DIR / "polymarket_ml_dataset_clean.parquet"
    )

    y_fit  = train_fit[TARGET].values.astype(np.int32)
    y_cal  = train_cal[TARGET].values.astype(np.int32)
    y_test = test[TARGET].values.astype(np.int32)

    X_fit_price  = train_fit[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_fit_full   = train_fit[FEATURES_FULL].values.astype(np.float32)
    X_cal_full   = train_cal[FEATURES_FULL].values.astype(np.float32)
    X_test_price = test[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_test_full  = test[FEATURES_FULL].values.astype(np.float32)

    sample_weights = make_sample_weights(y_fit, train_fit["market_id"].values)
    log.info("  Sample weight range: %.4f – %.4f", sample_weights.min(), sample_weights.max())

    best_params = run_grid_search(X_fit_full, y_fit, sample_weights, X_cal_full, y_cal)
    rf_params   = {**RF_BASE, **best_params}

    run_cv(X_fit_full, y_fit, train_fit["market_id"].values, sample_weights, best_params)

    log.info("\nTraining rf_price_only (300 trees, default depth=16 leaf=50) ...")
    rf_price = RandomForestClassifier(**RF_BASE, max_depth=16, min_samples_leaf=50)
    rf_price.fit(X_fit_price, y_fit, sample_weight=sample_weights)
    joblib.dump(rf_price, ARTIFACTS_DIR / "rf_price_only.pkl")
    log.info("  Saved rf_price_only.pkl")

    log.info("Training rf_full (300 trees, best params) ...")
    rf_full = RandomForestClassifier(**rf_params)
    rf_full.fit(X_fit_full, y_fit, sample_weight=sample_weights)
    joblib.dump(rf_full, ARTIFACTS_DIR / "rf_full.pkl")
    log.info("  Saved rf_full.pkl (params: %s)", best_params)

    log.info("Calibrating rf_full with isotonic regression ...")
    p_cal_raw = rf_full.predict_proba(X_cal_full)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)
    joblib.dump({"rf": rf_full, "iso": iso}, ARTIFACTS_DIR / "rf_full_calibrated.pkl")
    log.info("  Saved rf_full_calibrated.pkl")

    proba_price = rf_price.predict_proba(X_test_price)[:, 1]
    proba_full  = rf_full.predict_proba(X_test_full)[:, 1]
    proba_cal   = iso.transform(rf_full.predict_proba(X_test_full)[:, 1])

    threshold, _ = find_optimal_threshold(y_test, proba_cal)

    evaluate(y_test, proba_price, label="RF — price only",  threshold=threshold)
    evaluate(y_test, proba_full,  label="RF — full",        threshold=threshold)
    evaluate(y_test, proba_cal,   label="RF — calibrated",  threshold=threshold)
    evaluate_by_category(test, proba_cal, threshold=threshold)
    evaluate_by_volume_quintile(test, proba_cal, threshold=threshold)
    analyze_market_disagreements(test, proba_cal, threshold=threshold)

    log.info("\nCalibration (before vs after):")
    log.info("  Before:")
    check_calibration(y_test, proba_full)
    log.info("  After:")
    check_calibration(y_test, proba_cal)

    log.info("\nSHAP feature attribution (TreeExplainer, sample=5000) ...")
    try:
        import shap
        rng_shap  = np.random.default_rng(42)
        idx_shap  = rng_shap.choice(len(X_test_full), size=min(5000, len(X_test_full)), replace=False)
        explainer = shap.TreeExplainer(rf_full)
        shap_vals = explainer.shap_values(X_test_full[idx_shap])
        # shap_values returns [class0, class1] for classifiers
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        mean_abs  = np.abs(sv).mean(axis=0)
        ranked    = sorted(zip(FEATURES_FULL, mean_abs), key=lambda x: x[1], reverse=True)
        log.info("  Top features by mean |SHAP|:")
        for feat, score in ranked[:10]:
            bar = "█" * max(1, int(score * 300))
            log.info("    %+.4f  %-30s  %s", score, feat, bar)
    except Exception as e:
        log.warning("  SHAP skipped: %s", e)

    pred_df = test[["market_id", "snapshot_timestamp", TARGET, "split"]].copy()
    pred_df["proba_price_only"]      = proba_price.round(6)
    pred_df["proba_full"]            = proba_full.round(6)
    pred_df["proba_full_calibrated"] = proba_cal.round(6)
    out = PREDICTIONS_DIR / "test_predictions.csv"
    pred_df.to_csv(out, index=False)
    log.info("\nSaved %s rows → %s", f"{len(pred_df):,}", out)


if __name__ == "__main__":
    main()
