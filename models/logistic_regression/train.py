import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

from models.common.evaluation import (  # noqa: E402
    evaluate,
    evaluate_by_category,
    find_optimal_threshold,
)
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
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )),
    ])



def print_feature_importance(pipeline, top_n=20):
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocessor"]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    coefs = clf.coef_[0]
    importance = sorted(zip(all_feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'─'*50}")
    print(f"  Top {top_n} features by |coefficient|")
    print(f"{'─'*50}")
    for name, coef in importance[:top_n]:
        bar = "+" if coef > 0 else "-"
        print(f"  {bar} {abs(coef):6.3f}  {name}")
    print(f"{'─'*50}")


def run_grid_search(pipeline, X_train, y_train, sample_weights):
    param_grid = [
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1"],
            "clf__solver": ["liblinear"],
        },
    ]
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train, clf__sample_weight=sample_weights)
    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV AUC : {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    df = load_data()
    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    counts = train.groupby("market_id").size()
    sample_weights = train["market_id"].map(counts).rdiv(1).values

    print("\nRunning grid search for logistic regression...")
    pipeline = build_pipeline()
    pipeline = run_grid_search(pipeline, X_train, y_train, sample_weights)
    print("  Done.")

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_prob)
    print(f"\n  Optimal threshold (max F1): {optimal_threshold:.3f}  (F1={best_f1:.4f})")

    evaluate(y_test, y_prob, label="Logistic Regression", threshold=optimal_threshold)
    evaluate_by_category(test.reset_index(drop=True), y_prob, threshold=optimal_threshold)

    print_feature_importance(pipeline)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump({"pipeline": pipeline}, model_path)
    print(f"\nModel saved to {model_path}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = test[["market_id", "category", TARGET]].copy().reset_index(drop=True)
    pred_df["pred_prob"]  = y_prob
    pred_df["pred_label"] = (y_prob >= optimal_threshold).astype(int)
    preds_path = PREDICTIONS_DIR / "predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
