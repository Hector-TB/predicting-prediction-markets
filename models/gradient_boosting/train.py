import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, log_loss, precision_recall_curve, roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = pathlib.Path(__file__).parent / "artifacts"
PREDICTIONS_DIR = pathlib.Path(__file__).parent / "predictions"

NUMERIC_FEATURES = [
    "price_at_snapshot",
    "price_deviation_from_half",
    "days_before_close",
    "pct_lifetime_elapsed",
    "duration_days",
    "log_volume",
    "price_mean_7d", "price_volatility_7d", "price_min_7d", "price_max_7d",
    "price_change_7d", "price_range_7d", "price_trend_7d",
    "price_mean_14d", "price_volatility_14d", "price_min_14d", "price_max_14d",
    "price_change_14d", "price_range_14d", "price_trend_14d",
]
CATEGORICAL_FEATURES = ["category"]
TARGET = "outcome"


def load_data():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_DIR / "polymarket_ml_dataset.parquet")
    df["category"] = df["category"].fillna("other")
    df = df.dropna(subset=[TARGET])
    print(f"  Total rows: {len(df):,}  |  train: {(df['split']=='train').sum():,}  |  test: {(df['split']=='test').sum():,}")
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=0), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"),           CATEGORICAL_FEATURES),
    ])


def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(y_true, y_prob, label="", threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  PR-AUC    : {average_precision_score(y_true, y_prob):.4f}")
    print(f"  Log-loss  : {log_loss(y_true, y_prob):.4f}")
    print(f"  Brier     : {brier_score_loss(y_true, y_prob):.4f}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_true, y_pred):.4f}")
    print(f"{'─'*50}")


def evaluate_by_category(test_df, y_prob, threshold):
    print(f"\n{'─'*60}")
    print(f"  Per-category metrics (threshold={threshold:.3f})")
    print(f"  {'Category':<20} {'N':>7}  {'AUC':>6}  {'PR-AUC':>7}  {'F1':>6}  {'YES%':>6}")
    print(f"{'─'*60}")

    for cat in sorted(test_df["category"].unique()):
        mask = test_df["category"] == cat
        y_true_cat = test_df.loc[mask, TARGET].values
        y_prob_cat = y_prob[mask.values]

        if len(np.unique(y_true_cat)) < 2 or len(y_true_cat) < 10:
            continue

        y_pred_cat = (y_prob_cat >= threshold).astype(int)
        auc  = roc_auc_score(y_true_cat, y_prob_cat)
        pr   = average_precision_score(y_true_cat, y_prob_cat)
        f1   = f1_score(y_true_cat, y_pred_cat)
        yes  = y_true_cat.mean()
        print(f"  {cat:<20} {mask.sum():>7,}  {auc:>6.4f}  {pr:>7.4f}  {f1:>6.4f}  {yes:>6.1%}")

    print(f"{'─'*60}")


def check_calibration(y_true, y_prob, n_bins=10):
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    print(f"\n{'─'*50}")
    print(f"  Calibration (predicted → actual YES rate)")
    print(f"  {'Predicted':>10}  {'Actual':>10}  {'Diff':>8}")
    print(f"{'─'*50}")
    for pred, actual in zip(mean_predicted, fraction_of_positives):
        diff = actual - pred
        flag = "  <-- over-confident" if diff < -0.05 else ("  <-- under-confident" if diff > 0.05 else "")
        print(f"  {pred:>10.3f}  {actual:>10.3f}  {diff:>+8.3f}{flag}")
    print(f"{'─'*50}")


def print_feature_importance(preprocessor, clf, top_n=20):
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    importances = clf.feature_importances_
    importance = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*50}")
    print(f"  Top {top_n} features by gain importance")
    print(f"{'─'*50}")
    for name, score in importance[:top_n]:
        bar = "█" * max(1, int(score * 200))
        print(f"  {score:6.4f}  {name:<35}  {bar}")
    print(f"{'─'*50}")


def main():
    df = load_data()
    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    all_market_ids = train["market_id"].unique()
    rng = np.random.default_rng(42)
    cal_market_ids = set(rng.choice(all_market_ids, size=int(len(all_market_ids) * 0.2), replace=False))

    train_fit = train[~train["market_id"].isin(cal_market_ids)]
    train_cal = train[train["market_id"].isin(cal_market_ids)]

    print(f"  Train fit: {len(train_fit):,} rows  |  Train cal: {len(train_cal):,} rows")

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_fit,  y_fit  = train_fit[FEATURES], train_fit[TARGET].values
    X_cal,  y_cal  = train_cal[FEATURES], train_cal[TARGET].values
    X_test, y_test = test[FEATURES],      test[TARGET].values

    print("\nFitting preprocessor...")
    preprocessor = build_preprocessor()
    X_fit_t  = preprocessor.fit_transform(X_fit)
    X_cal_t  = preprocessor.transform(X_cal)
    X_test_t = preprocessor.transform(X_test)
    print("  Done.")

    scale_pos_weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    print("\nTraining XGBoost...")
    clf = XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(
        X_fit_t, y_fit,
        eval_set=[(X_cal_t, y_cal)],
        verbose=100,
    )
    print(f"  Best iteration: {clf.best_iteration}  |  Best score: {clf.best_score:.4f}")

    print("\nCalibrating with isotonic regression...")
    p_cal_raw = clf.predict_proba(X_cal_t)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal)
    print("  Done.")

    y_prob_raw = clf.predict_proba(X_test_t)[:, 1]
    y_prob_cal = iso.transform(y_prob_raw)

    optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_prob_cal)
    print(f"\n  Optimal threshold (max F1 on calibrated): {optimal_threshold:.3f}  (F1={best_f1:.4f})")

    evaluate(y_test, y_prob_raw, label="XGBoost — Uncalibrated", threshold=optimal_threshold)
    evaluate(y_test, y_prob_cal, label="XGBoost — Calibrated",   threshold=optimal_threshold)
    evaluate_by_category(test.reset_index(drop=True), y_prob_cal, threshold=optimal_threshold)

    print("\n  === Calibration before vs after ===")
    print("  -- Before --")
    check_calibration(y_test, y_prob_raw)
    print("  -- After --")
    check_calibration(y_test, y_prob_cal)

    print_feature_importance(preprocessor, clf)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump({"preprocessor": preprocessor, "clf": clf, "calibrator": iso}, model_path)
    print(f"\nModel saved to {model_path}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = test[["market_id", "category", TARGET]].copy().reset_index(drop=True)
    pred_df["pred_prob_raw"] = y_prob_raw
    pred_df["pred_prob"]     = y_prob_cal
    pred_df["pred_label"]    = (y_prob_cal >= optimal_threshold).astype(int)
    preds_path = PREDICTIONS_DIR / "predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
