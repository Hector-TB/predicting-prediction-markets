"""
models/logistic_regression/train.py
===================================
Logistic regression under the shared training protocol (ADR-024): settings
chosen on the time-ordered holdout, isotonic calibration and threshold from
the same holdout, test scored once.

Usage:
    python models/logistic_regression/train.py            # base features
    python models/logistic_regression/train.py --trends   # + Google Trends
"""

import argparse
import logging
import pathlib
import sys

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.evaluation import (  # noqa: E402
    check_calibration,
    compute_metrics,
    dataset_version,
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)
from models.common.training import fit_calibrator, load_dataset, market_weights, split_holdout  # noqa: E402

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

# (C, penalty, solver)
GRID = [(c, "l2", "lbfgs") for c in [0.01, 0.1, 1.0, 10.0, 100.0]] + \
       [(c, "l1", "liblinear") for c in [0.01, 0.1, 1.0, 10.0, 100.0]]


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def print_feature_importance(pipeline, top_n=20):
    clf = pipeline.named_steps["clf"]
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    all_feature_names = NUMERIC_FEATURES + list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    importance = sorted(zip(all_feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)

    log.info("\n%s\n  Top %d features by |coefficient|\n%s", "─" * 50, top_n, "─" * 50)
    for name, coef in importance[:top_n]:
        log.info("  %s %6.3f  %s", "+" if coef > 0 else "-", abs(coef), name)
    log.info("─" * 50)


def run_grid_search(X_fit, y_fit, w_fit, X_hold, y_hold) -> dict:
    """Pick C × penalty by AUC on the holdout (ADR-024)."""
    log.info("\nGrid search: %d combinations ...", len(GRID))
    best_auc, best_params = -1.0, {}
    for i, (c, penalty, solver) in enumerate(GRID, 1):
        params = {"clf__C": c, "clf__penalty": penalty, "clf__solver": solver}
        pipe = build_pipeline().set_params(**params)
        pipe.fit(X_fit, y_fit, clf__sample_weight=w_fit)
        auc = compute_metrics(y_hold, pipe.predict_proba(X_hold)[:, 1])["auc"]
        log.info("  [%2d/%d]  C=%-6g  %s  →  AUC=%.4f%s", i, len(GRID), c, penalty, auc, " *" if auc > best_auc else "")
        if auc > best_auc:
            best_auc, best_params = auc, params
    log.info("  Best: %s  →  AUC=%.4f", best_params, best_auc)
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trends", action="store_true",
                        help="add Google Trends features; writes *_trends outputs alongside the base model")
    args = parser.parse_args()
    suffix = "_trends" if args.trends else ""
    if args.trends:
        NUMERIC_FEATURES.extend(TREND_FEATURES)

    df = load_dataset(trends=args.trends)
    fit, holdout = split_holdout(df[df["split"] == "train"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    del df

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_fit,  y_fit  = fit[FEATURES],     fit[TARGET].values
    X_hold, y_hold = holdout[FEATURES], holdout[TARGET].values
    X_test, y_test = test[FEATURES],    test[TARGET].values
    w_fit = market_weights(fit["market_id"])

    best_params = run_grid_search(X_fit, y_fit, w_fit, X_hold, y_hold)

    log.info("\nTraining final model on fit markets ...")
    pipeline = build_pipeline().set_params(**best_params)
    pipeline.fit(X_fit, y_fit, clf__sample_weight=w_fit)

    log.info("Calibrating with isotonic regression on the holdout ...")
    p_hold_raw = pipeline.predict_proba(X_hold)[:, 1]
    iso = fit_calibrator(p_hold_raw, y_hold)

    # Threshold chosen on the holdout — never on the test set (ADR-021)
    optimal_threshold, best_f1 = find_optimal_threshold(y_hold, iso.transform(p_hold_raw))
    log.info("  Threshold (max F1 on holdout): %.3f  (F1=%.4f)", optimal_threshold, best_f1)

    y_prob_raw = pipeline.predict_proba(X_test)[:, 1]
    y_prob_cal = iso.transform(y_prob_raw)

    evaluate(y_test, y_prob_raw, label="Logistic Regression — Uncalibrated", threshold=optimal_threshold)
    evaluate(y_test, y_prob_cal, label="Logistic Regression — Calibrated",   threshold=optimal_threshold)
    evaluate_by_category(test, y_prob_cal, threshold=optimal_threshold)

    log.info("\n  === Calibration before vs after ===\n  -- Before --")
    check_calibration(y_test, y_prob_raw)
    log.info("  -- After --")
    check_calibration(y_test, y_prob_cal)

    print_feature_importance(pipeline)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"model{suffix}.joblib"
    joblib.dump({"pipeline": pipeline, "calibrator": iso, "threshold": optimal_threshold}, model_path)
    log.info("\nModel saved to %s", model_path)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = test[["market_id", "snapshot_timestamp", "category", TARGET]].copy()
    pred_df["pred_prob_raw"]   = y_prob_raw
    pred_df["pred_prob"]       = y_prob_cal
    pred_df["dataset_version"] = dataset_version()
    pred_df["pred_label"]      = (y_prob_cal >= optimal_threshold).astype(int)
    preds_path = PREDICTIONS_DIR / f"predictions{suffix}.csv"
    pred_df.to_csv(preds_path, index=False)
    log.info("Predictions saved to %s", preds_path)


if __name__ == "__main__":
    main()
