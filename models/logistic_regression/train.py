import pathlib
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    meta = pd.read_csv(DATA_DIR / "polymarket_markets_meta.csv",
                       usecols=["market_id", "category"])
    meta["market_id"] = meta["market_id"].astype(str)
    df["market_id"] = df["market_id"].astype(str)
    df = df.drop(columns=["category"], errors="ignore")
    df = df.merge(meta, on="market_id", how="left")
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
            n_jobs=-1,
        )),
    ])


def evaluate(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{'─'*40}")
    if label:
        print(f"  {label}")
    print(f"  AUC-ROC  : {roc_auc_score(y_true, y_prob):.4f}  (baseline: 0.9640)")
    print(f"  Log-loss : {log_loss(y_true, y_prob):.4f}  (baseline: 0.1710)")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}  (baseline: 0.9380)")
    print(f"  Brier    : {brier_score_loss(y_true, y_prob):.4f}")
    print(f"{'─'*40}")


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


def main():
    df = load_data()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    print("\nTraining logistic regression...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Done.")

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_prob, label="Logistic Regression — Test Set")
    print_feature_importance(pipeline)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = test[["condition_id", TARGET]].copy() if "condition_id" in test.columns else test[[TARGET]].copy()
    pred_df["pred_prob"] = y_prob
    pred_df["pred_label"] = (y_prob >= 0.5).astype(int)
    preds_path = PREDICTIONS_DIR / "predictions.csv"
    pred_df.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
