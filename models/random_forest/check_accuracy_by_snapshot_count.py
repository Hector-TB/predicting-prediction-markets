import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss

# ── LOAD DATA ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_PATH  = SCRIPT_DIR / "../../data/polymarket_ml_dataset.parquet"

print(f"Data path: {DATA_PATH.resolve()}")
print(f"File exists: {DATA_PATH.exists()}")

df = pd.read_parquet(DATA_PATH, engine="fastparquet")
df['category'] = df['category'].fillna('unknown')

train = df[df['split'] == 'train'].copy()
test  = df[df['split'] == 'test'].copy()

print(f"\nTrain: {train.shape[0]:,} rows, {train['market_id'].nunique():,} markets")
print(f"Test:  {test.shape[0]:,} rows,  {test['market_id'].nunique():,} markets")

# ── ORDINAL ENCODING ────────────────────────────────────────────────────────────
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train["category_encoded"] = enc.fit_transform(train[["category"]]).astype(np.float32)
test["category_encoded"]  = enc.transform(test[["category"]]).astype(np.float32)

# ── FEATURES AND PREDICTIONS ────────────────────────────────────────────────────
FEATURES_FULL = [
    "price_at_snapshot", "price_deviation_from_half",
    "price_mean_7d",     "price_volatility_7d",  "price_min_7d",
    "price_max_7d",      "price_change_7d",       "price_range_7d",  "price_trend_7d",
    "price_mean_14d",    "price_volatility_14d",  "price_min_14d",
    "price_max_14d",     "price_change_14d",      "price_range_14d", "price_trend_14d",
    "pct_lifetime_elapsed", "days_before_close", "duration_days",
    "log_volume", "category_encoded",
]

X_train = train[FEATURES_FULL]
y_train = train['outcome']
X_test  = test[FEATURES_FULL]
y_test  = test['outcome']

print("\nLoading rf_full_calibrated.pkl...")
rf_full_cal = joblib.load(SCRIPT_DIR / "artifacts/rf_full_calibrated.pkl")

# Generate predictions on both train and test
y_train_proba = rf_full_cal.predict_proba(X_train)[:, 1]
y_train_class = rf_full_cal.predict(X_train)
y_test_proba  = rf_full_cal.predict_proba(X_test)[:, 1]
y_test_class  = rf_full_cal.predict(X_test)

# ── OVERALL PERFORMANCE (train vs test) ─────────────────────────────────────────
# Train metrics diagnose overfitting — a large train/test gap means the model
# memorized training data rather than learning generalizable patterns.
train_auc   = roc_auc_score(y_train, y_train_proba)
train_brier = brier_score_loss(y_train, y_train_proba)
test_auc    = roc_auc_score(y_test, y_test_proba)
test_brier  = brier_score_loss(y_test, y_test_proba)

print(f"\n── Overall Performance ───────────────────────────────────")
print(f"{'Metric':<20} {'Train':>10} {'Test':>10} {'Gap':>10}")
print(f"{'-'*52}")
print(f"{'ROC-AUC':<20} {train_auc:>10.4f} {test_auc:>10.4f} {test_auc - train_auc:>+10.4f}")
print(f"{'Brier Score':<20} {train_brier:>10.4f} {test_brier:>10.4f} {test_brier - train_brier:>+10.4f}")

# ── HELPER: SNAPSHOT BIN ASSIGNMENT ─────────────────────────────────────────────
BINS   = [0, 50, 100, 200, 400, 1300]
LABELS = ['1-50', '51-100', '101-200', '201-400', '401+']

def assign_snapshot_bins(df_subset):
    market_counts = df_subset.groupby('market_id').size().reset_index(name='n_snapshots')
    market_counts['snapshot_bin'] = pd.cut(
        market_counts['n_snapshots'], bins=BINS, labels=LABELS
    )
    return market_counts

# ── SECTION 1: TRAIN BIAS ────────────────────────────────────────────────────────
# Question: is the model being trained predominantly on easy (long) markets?
# A market with 1,000 snapshots contributes 1,000 gradient signals vs 1 for a
# market with 1 snapshot — if long markets are systematically easier, the model
# is implicitly optimized toward easy cases.

print(f"\n── Section 1: Train Bias ─────────────────────────────────")
print(f"Are longer markets (more snapshots) dominating the training set?\n")

train_market_stats = train.groupby('market_id').agg(
    n_snapshots  = ('market_id', 'size'),
    outcome      = ('outcome', 'first'),
    duration     = ('duration_days', 'first')
).reset_index()

train_market_stats['snapshot_bin'] = pd.cut(
    train_market_stats['n_snapshots'], bins=BINS, labels=LABELS
)

train_summary = train_market_stats.groupby('snapshot_bin', observed=True).agg(
    n_markets    = ('market_id', 'count'),
    total_rows   = ('n_snapshots', 'sum'),
    avg_duration = ('duration', 'mean'),
    outcome_rate = ('outcome', 'mean')
).reset_index()

train_summary['pct_markets'] = (train_summary['n_markets'] / train_summary['n_markets'].sum()).round(3)
train_summary['pct_rows']    = (train_summary['total_rows'] / train_summary['total_rows'].sum()).round(3)

print(train_summary[[
    'snapshot_bin', 'n_markets', 'pct_markets',
    'total_rows', 'pct_rows', 'avg_duration', 'outcome_rate'
]].round(3).to_string(index=False))

print(f"\nTotal train rows:    {train_summary['total_rows'].sum():,}")
print(f"Total train markets: {train_summary['n_markets'].sum():,}")
print(f"\nInterpretation: compare pct_markets vs pct_rows per bin.")
print(f"If pct_rows >> pct_markets for the 401+ bin, long markets dominate training.")

# ── SECTION 2: TEST BIAS ─────────────────────────────────────────────────────────
# Question: are our reported test metrics inflated by longer/easier markets?
# Even with an unbiased model, if longer markets are easier to predict AND
# contribute more test rows, aggregate metrics will look better than they
# would on a balanced evaluation set.

print(f"\n── Section 2: Test Bias ──────────────────────────────────")
print(f"Are reported test metrics inflated by longer markets?\n")

test_eval = test.copy()
test_eval['pred_class'] = y_test_class
test_eval['pred_proba'] = y_test_proba
test_eval['correct']    = (test_eval['pred_class'] == test_eval['outcome']).astype(int)

market_accuracy = test_eval.groupby('market_id').agg(
    n_snapshots  = ('market_id', 'size'),
    accuracy     = ('correct', 'mean'),
    brier        = ('pred_proba', lambda x: np.mean(
                        (x - test_eval.loc[x.index, 'outcome'])**2)),
    outcome      = ('outcome', 'first'),
).reset_index()

market_accuracy['snapshot_bin'] = pd.cut(
    market_accuracy['n_snapshots'], bins=BINS, labels=LABELS
)

# Correlation check
corr_accuracy = market_accuracy['n_snapshots'].corr(market_accuracy['accuracy'])
corr_brier    = market_accuracy['n_snapshots'].corr(market_accuracy['brier'])

print(f"Correlation: snapshot count vs market accuracy: {corr_accuracy:+.3f}")
print(f"Correlation: snapshot count vs market brier:    {corr_brier:+.3f}")

# Breakdown by bin — both market count and row contribution
test_summary = market_accuracy.groupby('snapshot_bin', observed=True).agg(
    n_markets    = ('market_id', 'count'),
    total_rows   = ('n_snapshots', 'sum'),
    avg_accuracy = ('accuracy', 'mean'),
    avg_brier    = ('brier', 'mean'),
    outcome_rate = ('outcome', 'mean')
).reset_index()

test_summary['pct_markets'] = (test_summary['n_markets'] / test_summary['n_markets'].sum()).round(3)
test_summary['pct_rows']    = (test_summary['total_rows'] / test_summary['total_rows'].sum()).round(3)

print(f"\n{test_summary.round(3).to_string(index=False)}")
print(f"\nInterpretation: if pct_rows >> pct_markets for high-accuracy bins,")
print(f"aggregate metrics are pulled toward easier markets.")