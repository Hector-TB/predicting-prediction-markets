"""
models/gradient_boosting/train.py
=================================
XGBoost under the shared training protocol (ADR-024): settings and the number
of trees chosen on the time-ordered holdout, isotonic calibration and threshold
from the same holdout, test scored once.

Usage:
    python models/gradient_boosting/train.py            # base features
    python models/gradient_boosting/train.py --trends   # + Google Trends
"""

import argparse
import itertools
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.evaluation import (  # noqa: E402
    check_calibration,
    dataset_version,
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)
from models.common.training import (  # noqa: E402
    add_training_args, fit_calibrator, load_dataset, load_params, market_weights, split_holdout,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = pathlib.Path(__file__).parent / "predictions"

NUMERIC_FEATURES = [
    "price_at_snapshot",
    "price_deviation_from_half",
    "days_before_close",
    "pct_lifetime_elapsed",
    "duration_days",
    # log_volume dropped: it is the final lifetime volume, unknown at snapshot time (ADR-018)
    "price_mean_7d", "price_volatility_7d", "price_min_7d", "price_max_7d",
    "price_change_7d", "price_range_7d", "price_trend_7d",
    "price_mean_14d", "price_volatility_14d", "price_min_14d", "price_max_14d",
    "price_change_14d", "price_range_14d", "price_trend_14d",
]
CATEGORICAL_FEATURES = ["category"]
# Google Trends features (ADR-009) — same set as random_forest_trends
TREND_FEATURES = ["trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data"]
TARGET = "outcome"

GRID = {
    "learning_rate":    [0.01, 0.05, 0.1],
    "max_depth":        [4, 6, 8],
    "min_child_weight": [5, 10, 20],
}


def predict(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Calibrated P(YES) for each row of `df` from a saved artifact (used by releases, ADR-019)."""
    features = artifact.get("features", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    X = df[features].assign(category=df["category"].fillna("other"))
    raw = artifact["clf"].predict_proba(artifact["preprocessor"].transform(X))[:, 1]
    return artifact["calibrator"].transform(raw)


def build_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=0), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"),           CATEGORICAL_FEATURES),
    ])


def make_classifier(scale_pos_weight: float, **params) -> XGBClassifier:
    """Fixed settings plus the tuned ones; given settings (e.g. n_estimators) override the defaults."""
    return XGBClassifier(**{
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        **params,
    })


def print_feature_importance(preprocessor, clf, top_n=20):
    cat_encoder = preprocessor.named_transformers_["cat"]
    all_feature_names = NUMERIC_FEATURES + list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    importance = sorted(zip(all_feature_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)

    log.info("\n%s\n  Top %d features by gain importance\n%s", "─" * 50, top_n, "─" * 50)
    for name, score in importance[:top_n]:
        log.info("  %6.4f  %-35s  %s", score, name, "█" * max(1, int(score * 200)))
    log.info("─" * 50)


def run_grid_search(X_fit_t, y_fit, w_fit, X_hold_t, y_hold, scale_pos_weight) -> dict:
    """Pick settings by AUC on the holdout; early stopping on the same holdout sets the tree count (ADR-024)."""
    combos = list(itertools.product(GRID["learning_rate"], GRID["max_depth"], GRID["min_child_weight"]))
    log.info("\nGrid search: %d combinations ...", len(combos))

    best_auc, best_params = -1.0, {}
    for i, (lr, depth, mcw) in enumerate(combos, 1):
        clf = make_classifier(scale_pos_weight, learning_rate=lr, max_depth=depth, min_child_weight=mcw)
        clf.fit(X_fit_t, y_fit, sample_weight=w_fit, eval_set=[(X_hold_t, y_hold)], verbose=False)
        auc = clf.best_score
        log.info("  [%2d/%d]  lr=%-5g depth=%d  mcw=%-3d →  AUC=%.4f  (iters=%d)%s",
                 i, len(combos), lr, depth, mcw, auc, clf.best_iteration, " *" if auc > best_auc else "")
        if auc > best_auc:
            best_auc, best_params = auc, {"learning_rate": lr, "max_depth": depth, "min_child_weight": mcw}

    log.info("  Best: %s  →  AUC=%.4f", best_params, best_auc)
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trends", action="store_true",
                        help="add Google Trends features; writes *_trends outputs alongside the base model")
    add_training_args(parser)
    args = parser.parse_args()
    # Given settings include the tree count, so no early stopping either
    params = load_params(args.params, {"learning_rate", "max_depth", "min_child_weight", "n_estimators"})
    artifacts_dir = args.out_dir / "artifacts" if args.out_dir else ARTIFACTS_DIR
    predictions_dir = args.out_dir / "predictions" if args.out_dir else PREDICTIONS_DIR
    suffix = "_trends" if args.trends else ""
    if args.trends:
        NUMERIC_FEATURES.extend(TREND_FEATURES)

    df = load_dataset(trends=args.trends)
    fit, holdout = split_holdout(df[df["split"] == "train"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    del df

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y_fit, y_hold, y_test = fit[TARGET].values, holdout[TARGET].values, test[TARGET].values
    w_fit = market_weights(fit["market_id"])

    log.info("\nFitting preprocessor ...")
    preprocessor = build_preprocessor()
    X_fit_t  = preprocessor.fit_transform(fit[FEATURES])
    X_hold_t = preprocessor.transform(holdout[FEATURES])
    X_test_t = preprocessor.transform(test[FEATURES])

    scale_pos_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    log.info("  scale_pos_weight: %.2f", scale_pos_weight)

    if params:
        log.info("\nTraining XGBoost with the given settings (%d trees, no early stopping) ...", params["n_estimators"])
        clf = make_classifier(scale_pos_weight, **params)
        clf.set_params(early_stopping_rounds=None)
        clf.fit(X_fit_t, y_fit, sample_weight=w_fit, verbose=False)
        saved_params = params
    else:
        best_params = run_grid_search(X_fit_t, y_fit, w_fit, X_hold_t, y_hold, scale_pos_weight)
        log.info("\nTraining XGBoost with best params ...")
        clf = make_classifier(scale_pos_weight, **best_params)
        clf.fit(X_fit_t, y_fit, sample_weight=w_fit, eval_set=[(X_hold_t, y_hold)], verbose=100)
        log.info("  Best iteration: %d  |  Holdout AUC: %.4f", clf.best_iteration, clf.best_score)
        saved_params = {**best_params, "n_estimators": clf.best_iteration + 1}

    log.info("Calibrating with isotonic regression on the holdout ...")
    p_hold_raw = clf.predict_proba(X_hold_t)[:, 1]
    iso = fit_calibrator(p_hold_raw, y_hold)

    # Threshold chosen on the holdout — never on the test set (ADR-021)
    optimal_threshold, best_f1 = find_optimal_threshold(y_hold, iso.transform(p_hold_raw))
    log.info("  Threshold (max F1 on holdout): %.3f  (F1=%.4f)", optimal_threshold, best_f1)

    y_prob_raw = clf.predict_proba(X_test_t)[:, 1]
    y_prob_cal = iso.transform(y_prob_raw)

    evaluate(y_test, y_prob_raw, label="XGBoost — Uncalibrated", threshold=optimal_threshold)
    evaluate(y_test, y_prob_cal, label="XGBoost — Calibrated",   threshold=optimal_threshold)
    evaluate_by_category(test, y_prob_cal, threshold=optimal_threshold)

    log.info("\n  === Calibration before vs after ===\n  -- Before --")
    check_calibration(y_test, y_prob_raw)
    log.info("  -- After --")
    check_calibration(y_test, y_prob_cal)

    print_feature_importance(preprocessor, clf)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"model{suffix}.joblib"
    joblib.dump({"preprocessor": preprocessor, "clf": clf, "calibrator": iso, "threshold": optimal_threshold,
                 "features": FEATURES, "params": saved_params},
                model_path)
    log.info("\nModel saved to %s", model_path)

    predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_df = test[["market_id", "snapshot_timestamp", "category", TARGET]].copy()
    pred_df["pred_prob_raw"]   = y_prob_raw
    pred_df["pred_prob"]       = y_prob_cal
    pred_df["dataset_version"] = dataset_version()
    pred_df["pred_label"]      = (y_prob_cal >= optimal_threshold).astype(int)
    preds_path = predictions_dir / f"predictions{suffix}.csv"
    pred_df.to_csv(preds_path, index=False)
    log.info("Predictions saved to %s", preds_path)


if __name__ == "__main__":
    main()
