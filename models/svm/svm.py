import pandas as pd
import numpy as np
import joblib
import time
import warnings
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

SCRIPT_DIR    = Path(__file__).parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE  = 42

DATA_PATH = SCRIPT_DIR / "../../data/polymarket_ml_dataset_with_trends_clean.parquet"

PERCENTILE_POINTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Best params from grid search — consistent across multiple runs
BEST_C     = 10.0
BEST_GAMMA = 0.001

# ── SECTION 1: LOAD DATA AND FIXED PERCENTILE SUBSAMPLING ───────────────────────
# Rather than keeping all ~187 snapshots per market (which creates training bias
# toward long markets and near-duplicate consecutive rows), we retain only
# snapshots at standardized percentiles of each market's lifetime.
#
# This gives us:
#   - Exactly 7 rows per market → every market contributes equally to training
#   - Genuinely distinct observations (20% vs 30% elapsed is a meaningful gap)
#   - Dataset shrinks from 4M to ~150k rows → SVM is computationally feasible
#   - Same prediction task: "at snapshot T, predict final outcome"
#   - All existing snapshot-level features remain valid
#
# For each percentile point we find the snapshot whose pct_lifetime_elapsed
# is closest to the target — we never interpolate or fabricate data.

print("\n" + "="*60)
print("SECTION 1: LOAD DATA AND FIXED PERCENTILE SUBSAMPLING")
print("="*60)

print(f"Loading {DATA_PATH.name}...")
df = pd.read_parquet(DATA_PATH)
df['category'] = df['category'].fillna('unknown')
df['outcome']  = df['outcome'].astype(int)
df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'], format='ISO8601', utc=True)

print(f"Full dataset: {df.shape[0]:,} rows, {df['market_id'].nunique():,} markets")
print(f"Split balance: {df.groupby('split')['market_id'].nunique().to_dict()}")

def get_percentile_snapshots(group):
    """
    For a single market's snapshot history, find the snapshot closest to
    each target percentile of lifetime elapsed. Returns exactly
    len(PERCENTILE_POINTS) rows per market, one per percentile point.
    """
    group = group.sort_values('pct_lifetime_elapsed')
    selected = []
    for pct in PERCENTILE_POINTS:
        idx = (group['pct_lifetime_elapsed'] - pct).abs().idxmin()
        row = group.loc[idx].copy()
        row['target_percentile'] = pct
        selected.append(row)
    return pd.DataFrame(selected)

print(f"\nSubsampling to {len(PERCENTILE_POINTS)} fixed percentile snapshots per market...")
print(f"Target percentiles: {[f'{p:.0%}' for p in PERCENTILE_POINTS]}")

df_sampled = (
    df.groupby('market_id', group_keys=False)
    .apply(get_percentile_snapshots)
    .reset_index(drop=True)
)

rows_per_market = df_sampled.groupby('market_id').size()

print(f"\nDataset after subsampling:")
print(f"  Rows:    {df_sampled.shape[0]:,}  (was {df.shape[0]:,})")
print(f"  Markets: {df_sampled['market_id'].nunique():,}")
print(f"  Rows per market — min: {rows_per_market.min()}, max: {rows_per_market.max()}, mean: {rows_per_market.mean():.1f}")
print(f"  All markets have exactly {len(PERCENTILE_POINTS)} rows: {(rows_per_market == len(PERCENTILE_POINTS)).all()}")

print(f"\nActual vs target pct_lifetime_elapsed (mean across markets):")
actual_pcts = df_sampled.groupby('target_percentile')['pct_lifetime_elapsed'].mean()
for target, actual in actual_pcts.items():
    print(f"  target={target:.2f}  actual={actual:.3f}  diff={abs(actual-target):.3f}")

print(f"\nOutcome balance: {df_sampled['outcome'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Split balance:   {df_sampled['split'].value_counts().to_dict()}")

# Save single subsampled dataset — both models draw from this
sampled_path = ARTIFACTS_DIR / "df_sampled_clean.parquet"
df_sampled.to_parquet(sampled_path, index=False)
print(f"\nSaved subsampled dataset to: {sampled_path}")
print(f"File size: {sampled_path.stat().st_size / (1024*1024):.1f} MB")

# ── SECTION 2: PREPARE TRAIN/TEST SPLITS ────────────────────────────────────────
# Section 1 only needs to run once to generate df_sampled_clean.parquet.
# After that, comment out section 1 and load directly from here.

print("\n" + "="*60)
print("SECTION 2: PREPARE TRAIN/TEST SPLITS")
print("="*60)

df_sampled = pd.read_parquet(ARTIFACTS_DIR / "df_sampled_clean.parquet")
df_sampled['outcome']  = df_sampled['outcome'].astype(int)
df_sampled['category'] = df_sampled['category'].fillna('unknown')

assert df_sampled['outcome'].isna().sum() == 0, "NAs in outcome"
assert (df_sampled.groupby('market_id').size() == 7).all(), "Not all markets have 7 rows"

# Full split — for non-trends model (all markets including 'other' category)
train_all = df_sampled[df_sampled['split'] == 'train'].copy().reset_index(drop=True)
test_all  = df_sampled[df_sampled['split'] == 'test'].copy().reset_index(drop=True)

print(f"Full split (all markets):")
print(f"  Train: {train_all.shape[0]:,} rows, {train_all['market_id'].nunique():,} markets")
print(f"  Test:  {test_all.shape[0]:,} rows,  {test_all['market_id'].nunique():,} markets")

# Trends subset — exclude markets with no trend data
# The 'other' category has zero trend coverage; also ~133 miscellaneous markets
markets_with_trends = df_sampled.groupby('market_id')['has_trend_data'].max()
trend_market_ids    = markets_with_trends[markets_with_trends == 1].index

train_trends = train_all[train_all['market_id'].isin(trend_market_ids)].copy().reset_index(drop=True)
test_trends  = test_all[test_all['market_id'].isin(trend_market_ids)].copy().reset_index(drop=True)

print(f"\nTrends subset (markets with trend data):")
print(f"  Train: {train_trends.shape[0]:,} rows, {train_trends['market_id'].nunique():,} markets")
print(f"  Test:  {test_trends.shape[0]:,} rows,  {test_trends['market_id'].nunique():,} markets")
print(f"  Markets dropped (no trend data): {train_all['market_id'].nunique() - train_trends['market_id'].nunique():,}")
print(f"  Categories in trends subset: {sorted(train_trends['category'].unique())}")

# ── SECTION 3: DEFINE FEATURE SETS ──────────────────────────────────────────────
# Base features — shared by both models
#
# EXCLUDED features and reasons:
#   market_id                 — unique identifier, not generalizable
#   snapshot_timestamp        — raw time; target_percentile captures position
#   price_deviation_from_half — mathematically redundant with price_at_snapshot
#   total_volume              — redundant with log_volume
#   days_before_close         — redundant with target_percentile + duration_days
#   price_mean_7d/14d         — r>0.96 with price_at_snapshot, nearly redundant
#   pct_lifetime_elapsed      — r=0.93 with target_percentile; target is cleaner
#   price_min/max_7d/14d      — captured by mean + volatility
#   price_range_7d/14d        — captured by volatility
#   has_trend_data            — redundant with category (other=0, else=1)
#   outcome                   — target variable
#   split                     — metadata
#   question                  — raw text, needs NLP

print("\n" + "="*60)
print("SECTION 3: DEFINE FEATURE SETS")
print("="*60)

base_numeric = [
    'price_at_snapshot',      # crowd probability estimate — strongest feature
    'target_percentile',      # standardized lifecycle stage (0.20-0.80)
    'duration_days',          # total market length
    'log_volume',             # liquidity proxy
    'price_volatility_7d',    # market uncertainty over past 7 days
    'price_change_7d',        # net price movement over past 7 days — momentum
    'price_trend_7d',         # slope of price over past 7 days — directional momentum
    'price_volatility_14d',   # market uncertainty over past 14 days
    'price_change_14d',       # net price movement over past 14 days
    'price_trend_14d',        # slope of price over past 14 days
]

trends_numeric = [
    'trend_value',            # google trends search interest (0-100 scale)
    'trend_ma4',              # 4-week moving average of trend_value
    'trend_change_4w',        # change in trend_value over past 4 weeks
    'trend_spike',            # binary: did search interest spike recently
]

categorical_features = ['category']
features_no_trends   = base_numeric + categorical_features
features_with_trends = base_numeric + trends_numeric + categorical_features
target               = 'outcome'

print(f"Non-trends model: {len(features_no_trends)} features ({len(base_numeric)} numeric + 1 categorical)")
print(f"Trends model:     {len(features_with_trends)} features ({len(base_numeric) + len(trends_numeric)} numeric + 1 categorical)")
print(f"Trends features added: {trends_numeric}")

# Correlation check — confirm no highly correlated pairs in base features
corr_matrix = train_all[base_numeric].corr().round(2)
high_corr   = []
for i in range(len(base_numeric)):
    for j in range(i+1, len(base_numeric)):
        c = abs(corr_matrix.iloc[i, j])
        if c > 0.85:
            high_corr.append((base_numeric[i], base_numeric[j], c))

print(f"\nHighly correlated pairs (|r| > 0.85): {len(high_corr)}")
if high_corr:
    for f1, f2, c in sorted(high_corr, key=lambda x: -x[2]):
        print(f"  {f1} vs {f2}  r={c:.2f}")
else:
    print("  None — feature set is clean")

print(f"\nCorrelations of trend features with price_at_snapshot:")
for col in trends_numeric:
    corr = train_trends['price_at_snapshot'].corr(train_trends[col])
    print(f"  {col:25s}  r={corr:+.3f}")

print(f"\nMissing values in trend features (train_trends):")
print(train_trends[trends_numeric].isna().sum().to_string())

# ── SECTION 4: HANDLE RARE CATEGORIES ───────────────────────────────────────────
# Collapse category levels appearing in fewer than 2% of markets into 'other'.
# Frequency computed at market level (not row level) since each market has
# exactly 7 rows by construction.

print("\n" + "="*60)
print("SECTION 4: HANDLE RARE CATEGORIES")
print("="*60)

RARE_THRESHOLD = 0.02

def collapse_rare_categories(train_df, test_df, threshold=RARE_THRESHOLD):
    """Collapse rare category levels into 'other' in both train and test."""
    freq = (
        train_df.drop_duplicates('market_id')[['market_id', 'category']]
        .groupby('category')['market_id'].count()
        .div(train_df['market_id'].nunique())
    )
    rare     = freq[freq < threshold].index.tolist()
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df['category'] = train_df['category'].replace(rare, 'other')
    test_df['category']  = test_df['category'].replace(rare, 'other')
    known = set(train_df['category'].unique())
    test_df['category'] = test_df['category'].apply(lambda x: x if x in known else 'other')
    return train_df, test_df, rare

train_all,    test_all,    rare_all    = collapse_rare_categories(train_all,    test_all)
train_trends, test_trends, rare_trends = collapse_rare_categories(train_trends, test_trends)

print(f"Non-trends model:")
print(f"  Rare categories collapsed: {rare_all}")
print(f"  Final levels: {sorted(train_all['category'].unique())}")
print(f"  Outcome rate by category:")
print(train_all.drop_duplicates('market_id').groupby('category')['outcome'].mean().sort_values(ascending=False).round(3).to_string())

print(f"\nTrends model:")
print(f"  Rare categories collapsed: {rare_trends}")
print(f"  Final levels: {sorted(train_trends['category'].unique())}")

# ── SECTION 5: BUILD PREPROCESSING PIPELINES ────────────────────────────────────
# ColumnTransformer applies different preprocessing to numeric and categorical.
#
# Numeric:      SimpleImputer (median) → StandardScaler
#   StandardScaler is critical for SVM — without it features with large ranges
#   dominate the kernel distance computation over small-range features.
#
# Categorical:  SimpleImputer (constant) → OneHotEncoder
#   drop='first' avoids perfect multicollinearity.
#   handle_unknown='ignore' zeros out unseen categories in test.
#
# The preprocessor lives inside the Pipeline so it is fit only on training data
# and never sees validation/test data during CV — no leakage.

print("\n" + "="*60)
print("SECTION 5: BUILD PREPROCESSING PIPELINES")
print("="*60)

def build_preprocessor(num_features, cat_features):
    """Build ColumnTransformer: scaling for numeric, OHE for categorical."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Sanity check both preprocessors
for name, num_feats, train_df in [
    ('No-trends',   base_numeric,                  train_all),
    ('With-trends', base_numeric + trends_numeric, train_trends),
]:
    prep = build_preprocessor(num_feats, categorical_features)
    X    = train_df[num_feats + categorical_features]
    Xt   = prep.fit_transform(X)
    n_num = len(num_feats)
    n_cat = Xt.shape[1] - n_num
    means = Xt[:, :n_num].mean(axis=0).round(3)
    stds  = Xt[:, :n_num].std(axis=0).round(3)
    print(f"{name}: input {X.shape} → transformed {Xt.shape} ({n_num} numeric + {n_cat} dummies)")
    print(f"  Post-scaling means (should be ~0): {means}")
    print(f"  Post-scaling stds  (should be ~1): {stds}")
    print()

# ── SECTION 6: TRAIN NON-TRENDS SVM ─────────────────────────────────────────────
# Uses all markets (including 'other' category).
# Best params from previous grid search: C=10.0, gamma=0.001.
# class_weight='balanced' handles 82/18 outcome imbalance automatically.

print("\n" + "="*60)
print("SECTION 6: TRAIN NON-TRENDS SVM")
print("="*60)

X_train_all = train_all[features_no_trends]
y_train_all = train_all[target]

svm_no_trends = Pipeline(steps=[
    ('preprocessor', build_preprocessor(base_numeric, categorical_features)),
    ('svm', SVC(
        kernel='rbf',
        C=BEST_C,
        gamma=BEST_GAMMA,
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

print(f"Training non-trends SVM:")
print(f"  Markets: {train_all['market_id'].nunique():,}")
print(f"  Rows:    {len(X_train_all):,}")
print(f"  Params:  C={BEST_C}, gamma={BEST_GAMMA}")

start = time.time()
svm_no_trends.fit(X_train_all, y_train_all)
elapsed = time.time() - start

n_support = svm_no_trends['svm'].n_support_
print(f"\nTraining completed in {elapsed/60:.1f} minutes")
print(f"Support vectors: {sum(n_support):,} ({sum(n_support)/len(X_train_all)*100:.1f}% of training rows)")
print(f"  Class 0: {n_support[0]:,}  Class 1: {n_support[1]:,}")

model_path = ARTIFACTS_DIR / "svm_no_trends.pkl"
joblib.dump(svm_no_trends, model_path)
print(f"\nSaved: svm_no_trends.pkl ({model_path.stat().st_size / (1024*1024):.1f} MB)")

# ── SECTION 7: TRAIN TRENDS SVM ─────────────────────────────────────────────────
# Uses only markets with trend data (excludes 'other' category + ~133 markets).
# Same params as non-trends model for comparability.

print("\n" + "="*60)
print("SECTION 7: TRAIN TRENDS SVM")
print("="*60)

X_train_trends = train_trends[features_with_trends]
y_train_trends = train_trends[target]

svm_with_trends = Pipeline(steps=[
    ('preprocessor', build_preprocessor(base_numeric + trends_numeric, categorical_features)),
    ('svm', SVC(
        kernel='rbf',
        C=BEST_C,
        gamma=BEST_GAMMA,
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

print(f"Training trends SVM:")
print(f"  Markets: {train_trends['market_id'].nunique():,}")
print(f"  Rows:    {len(X_train_trends):,}")
print(f"  Params:  C={BEST_C}, gamma={BEST_GAMMA}")

start = time.time()
svm_with_trends.fit(X_train_trends, y_train_trends)
elapsed = time.time() - start

n_support = svm_with_trends['svm'].n_support_
print(f"\nTraining completed in {elapsed/60:.1f} minutes")
print(f"Support vectors: {sum(n_support):,} ({sum(n_support)/len(X_train_trends)*100:.1f}% of training rows)")
print(f"  Class 0: {n_support[0]:,}  Class 1: {n_support[1]:,}")

model_path = ARTIFACTS_DIR / "svm_with_trends.pkl"
joblib.dump(svm_with_trends, model_path)
print(f"\nSaved: svm_with_trends.pkl ({model_path.stat().st_size / (1024*1024):.1f} MB)")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Artifacts saved to: {ARTIFACTS_DIR}")
print(f"  df_sampled_clean.parquet  — subsampled dataset for evaluation")
print(f"  svm_no_trends.pkl         — trained on all {train_all['market_id'].nunique():,} markets")
print(f"  svm_with_trends.pkl       — trained on {train_trends['market_id'].nunique():,} markets with trend data")