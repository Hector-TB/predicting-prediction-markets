"""
models/random_forest_trends/train.py
=====================================
Trains four Random Forest variants on the Google Trends-enriched dataset:
  - rf_price_only          — price_at_snapshot only
  - rf_full                — 20 engineered features (no trends)
  - rf_trends              — 25 features (full + 5 Google Trends columns)
  - rf_trends_calibrated   — rf_trends + isotonic calibration on the holdout

Same protocol and helpers as models/random_forest/train.py (ADR-024); the grid
search runs on the trends feature set. The extra Trends features are the
primary comparison point (RQ2, RQ3).

Usage:
    python models/random_forest_trends/train.py
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT          = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

sys.path.insert(0, str(ROOT))
from models.common.evaluation import (  # noqa: E402
    dataset_version,
    analyze_market_disagreements,
    bootstrap_auc_diff,
    check_calibration,
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)
from models.common.training import fit_calibrator, market_weights  # noqa: E402
from models.random_forest.train import (  # noqa: E402
    FEATURES_FULL,
    FEATURES_PRICE_ONLY,
    RF_BASE,
    TARGET,
    load_and_split,
    log_shap,
    run_grid_search,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FEATURES_TRENDS = FEATURES_FULL + [
    "trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data",
]


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    fit, holdout, test = load_and_split(trends=True)

    y_fit  = fit[TARGET].values.astype(np.int32)
    y_hold = holdout[TARGET].values.astype(np.int32)
    y_test = test[TARGET].values.astype(np.int32)

    X_fit_price   = fit[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_fit_full    = fit[FEATURES_FULL].values.astype(np.float32)
    X_fit_trends  = fit[FEATURES_TRENDS].values.astype(np.float32)
    X_hold_trends = holdout[FEATURES_TRENDS].values.astype(np.float32)
    X_test_price  = test[FEATURES_PRICE_ONLY].values.astype(np.float32)
    X_test_full   = test[FEATURES_FULL].values.astype(np.float32)
    X_test_trends = test[FEATURES_TRENDS].values.astype(np.float32)

    w_fit = market_weights(fit["market_id"])

    best_params = run_grid_search(X_fit_trends, y_fit, w_fit, X_hold_trends, y_hold)
    rf_params   = {**RF_BASE, **best_params}

    log.info("\nTraining rf_price_only (300 trees, default depth=16 leaf=50) ...")
    rf_price = RandomForestClassifier(**RF_BASE, max_depth=16, min_samples_leaf=50)
    rf_price.fit(X_fit_price, y_fit, sample_weight=w_fit)
    joblib.dump(rf_price, ARTIFACTS_DIR / "trends_rf_price_only.pkl")

    log.info("Training rf_full (300 trees, best params) ...")
    rf_full = RandomForestClassifier(**rf_params)
    rf_full.fit(X_fit_full, y_fit, sample_weight=w_fit)
    joblib.dump(rf_full, ARTIFACTS_DIR / "trends_rf_full.pkl")

    log.info("Training rf_trends (300 trees, best params) ...")
    rf_trends = RandomForestClassifier(**rf_params)
    rf_trends.fit(X_fit_trends, y_fit, sample_weight=w_fit)
    joblib.dump(rf_trends, ARTIFACTS_DIR / "trends_rf_trends.pkl")
    log.info("  Saved trends_rf_trends.pkl (params: %s)", best_params)

    log.info("Calibrating rf_trends with isotonic regression on the holdout ...")
    p_hold_raw = rf_trends.predict_proba(X_hold_trends)[:, 1]
    iso = fit_calibrator(p_hold_raw, y_hold)

    # Threshold chosen on the holdout — never on the test set (ADR-021)
    threshold, _ = find_optimal_threshold(y_hold, iso.transform(p_hold_raw))
    joblib.dump({"rf": rf_trends, "iso": iso, "threshold": threshold},
                ARTIFACTS_DIR / "trends_rf_trends_calibrated.pkl")

    proba_price = rf_price.predict_proba(X_test_price)[:, 1]
    proba_full  = rf_full.predict_proba(X_test_full)[:, 1]
    proba_tr    = rf_trends.predict_proba(X_test_trends)[:, 1]
    proba_cal   = iso.transform(proba_tr)

    evaluate(y_test, proba_price, label="RF Trends — price only", threshold=threshold)
    evaluate(y_test, proba_full,  label="RF Trends — full",       threshold=threshold)
    evaluate(y_test, proba_tr,    label="RF Trends — trends",     threshold=threshold)
    evaluate(y_test, proba_cal,   label="RF Trends — calibrated", threshold=threshold)
    evaluate_by_category(test, proba_cal, threshold=threshold)
    analyze_market_disagreements(test, proba_cal, threshold=threshold)

    log.info("\nBootstrap CI: RF+Trends vs RF (full features, no trends) ...")
    bootstrap_auc_diff(y_test, proba_full, proba_tr, groups=test["market_id"].values)

    log.info("\nCalibration (before vs after):")
    log.info("  Before:")
    check_calibration(y_test, proba_tr)
    log.info("  After:")
    check_calibration(y_test, proba_cal)

    log_shap(rf_trends, X_test_trends, FEATURES_TRENDS, top_n=12)

    pred_df = test[["market_id", "snapshot_timestamp", TARGET, "split"]].copy()
    pred_df["dataset_version"]         = dataset_version()
    pred_df["proba_price_only"]        = proba_price.round(6)
    pred_df["proba_full"]              = proba_full.round(6)
    pred_df["proba_trends"]            = proba_tr.round(6)
    pred_df["proba_trends_calibrated"] = proba_cal.round(6)
    out = PREDICTIONS_DIR / "trends_test_predictions.csv"
    pred_df.to_csv(out, index=False)
    log.info("\nSaved %s rows → %s", f"{len(pred_df):,}", out)


if __name__ == "__main__":
    main()
