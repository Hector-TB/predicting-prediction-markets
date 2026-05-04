import pandas as pd
import numpy as np
from pathlib import Path

# ── 1. LOAD DATA AND FIXED PERCENTILE SUBSAMPLING ───────────────────────────────
# Rather than keeping all ~187 snapshots per market (which creates training bias
# toward long markets and near-duplicate consecutive rows), we retain only
# snapshots at standardized percentiles of each market's lifetime.
#
# This gives us:
#   - Exactly 9 rows per market → every market contributes equally to training
#   - Genuinely distinct observations (10% vs 20% elapsed is a meaningful gap)
#   - Dataset shrinks from 4M to ~153k rows → SVM is computationally feasible
#   - Same prediction task: "at snapshot T, predict final outcome"
#   - All existing snapshot-level features remain valid
#
# For each percentile point we find the snapshot whose pct_lifetime_elapsed
# is closest to the target — we never interpolate or fabricate data.

print("\n" + "="*60)
print("SECTION 1: LOAD DATA AND FIXED PERCENTILE SUBSAMPLING")
print("="*60)

PERCENTILE_POINTS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
RANDOM_STATE      = 42

SCRIPT_DIR = Path(__file__).parent
DATA_PATH  = SCRIPT_DIR / "../../data/polymarket_ml_dataset.parquet"

print(f"Data path:   {DATA_PATH.resolve()}")
print(f"File exists: {DATA_PATH.exists()}")

df = pd.read_parquet(DATA_PATH, engine="fastparquet")
df['category'] = df['category'].fillna('unknown')
df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'], format='ISO8601', utc=True)

print(f"\nFull dataset: {df.shape[0]:,} rows, {df['market_id'].nunique():,} markets")

# ── Subsample to fixed percentile snapshots ──────────────────────────────────────
def get_percentile_snapshots(group):
    """
    For a single market's snapshot history, find the snapshot closest to
    each target percentile of lifetime elapsed. Returns exactly len(PERCENTILE_POINTS)
    rows per market, one per percentile point.
    """
    group = group.sort_values('pct_lifetime_elapsed')
    selected = []
    for pct in PERCENTILE_POINTS:
        # Find the snapshot whose pct_lifetime_elapsed is closest to target pct
        idx = (group['pct_lifetime_elapsed'] - pct).abs().idxmin()
        row = group.loc[idx].copy()
        row['target_percentile'] = pct  # store which percentile this row represents
        selected.append(row)
    return pd.DataFrame(selected)

print(f"\nSubsampling to {len(PERCENTILE_POINTS)} fixed percentile snapshots per market...")
print(f"Target percentiles: {[f'{p:.0%}' for p in PERCENTILE_POINTS]}")

# df_sampled = (
#     df.groupby('market_id', group_keys=False)
#     .apply(get_percentile_snapshots)
#     .reset_index(drop=True)
# )

# # ── Sanity checks ────────────────────────────────────────────────────────────────
# rows_per_market = df_sampled.groupby('market_id').size()

# print(f"\nDataset after subsampling:")
# print(f"  Rows:    {df_sampled.shape[0]:,}  (was {df.shape[0]:,})")
# print(f"  Markets: {df_sampled['market_id'].nunique():,}")
# print(f"  Rows per market — min: {rows_per_market.min()}, max: {rows_per_market.max()}, mean: {rows_per_market.mean():.1f}")
# print(f"  All markets have exactly {len(PERCENTILE_POINTS)} rows: {(rows_per_market == len(PERCENTILE_POINTS)).all()}")

# # Confirm target percentiles are well-approximated
# print(f"\nActual vs target pct_lifetime_elapsed (mean across markets):")
# actual_pcts = df_sampled.groupby('target_percentile')['pct_lifetime_elapsed'].mean()
# for target, actual in actual_pcts.items():
#     print(f"  target={target:.2f}  actual={actual:.3f}  diff={abs(actual-target):.3f}")

# # Outcome and split balance
# print(f"\nOutcome balance: {df_sampled['outcome'].value_counts(normalize=True).round(3).to_dict()}")
# print(f"Split balance:   {df_sampled['split'].value_counts().to_dict()}")

# # ── Save subsampled dataset ──────────────────────────────────────────────────────
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
# ARTIFACTS_DIR.mkdir(exist_ok=True)  # create artifacts folder if it doesn't exist

# sampled_path = ARTIFACTS_DIR / "df_sampled.parquet"
# df_sampled.to_parquet(sampled_path, index=False)
# print(f"\nSaved subsampled dataset to: {sampled_path}")
# print(f"File size: {sampled_path.stat().st_size / (1024*1024):.1f} MB")

# ── 2. LOAD SUBSAMPLED DATA AND PREPARE SPLITS ──────────────────────────────────
# Section 1 only needs to run once to generate df_sampled.parquet.
# Comment out section 1 after first run and load directly from here.
#
# We also do basic cleaning here — parsing timestamps, ensuring outcome is
# integer 0/1, and filling NA categories — before splitting into train and test.

print("\n" + "="*60)
print("SECTION 2: LOAD SUBSAMPLED DATA AND PREPARE SPLITS")
print("="*60)

# Load subsampled dataset
df_sampled = pd.read_parquet(ARTIFACTS_DIR / "df_sampled.parquet")
print(f"Loaded subsampled dataset: {df_sampled.shape[0]:,} rows, {df_sampled['market_id'].nunique():,} markets")

# Basic cleaning
df_sampled['snapshot_timestamp'] = pd.to_datetime(df_sampled['snapshot_timestamp'], format='ISO8601', utc=True)
df_sampled['outcome']  = df_sampled['outcome'].astype(int)
df_sampled['category'] = df_sampled['category'].fillna('unknown')

assert df_sampled['outcome'].isna().sum() == 0, "NAs in outcome"
assert (df_sampled.groupby('market_id').size() == 7).all(), "Not all markets have 7 rows"

# Train/test split — pre-defined at market level, no leakage
train = df_sampled[df_sampled['split'] == 'train'].copy().reset_index(drop=True)
test  = df_sampled[df_sampled['split'] == 'test'].copy().reset_index(drop=True)

print(f"\nTrain: {train.shape[0]:,} rows, {train['market_id'].nunique():,} markets")
print(f"Test:  {test.shape[0]:,} rows,  {test['market_id'].nunique():,} markets")
print(f"\nRows per market — train: {train.groupby('market_id').size().mean():.1f}, test: {test.groupby('market_id').size().mean():.1f}")
print(f"\nOutcome balance — train: {train['outcome'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Outcome balance — test:  {test['outcome'].value_counts(normalize=True).round(3).to_dict()}")
print(f"\nCategory levels: {sorted(df_sampled['category'].unique())}")
print(f"Missing values:\n{df_sampled[['outcome', 'category', 'pct_lifetime_elapsed', 'price_at_snapshot']].isna().sum().to_string()}")

# ── 3. DEFINE FEATURES ──────────────────────────────────────────────────────────
# Same feature set as the original SVM implementation with one addition:
# target_percentile — which percentile of lifetime this snapshot represents
# (0.20, 0.30, ... 0.80). This replaces pct_lifetime_elapsed as the primary
# temporal position feature since by construction all our snapshots sit at
# standardized percentile points. Including it lets the model learn that
# the same price means different things at different lifecycle stages.
#
# EXCLUDED (same reasoning as before plus correlation findings from first run):
#   market_id              — unique identifier, not generalizable
#   snapshot_timestamp     — raw time not meaningful; target_percentile captures position
#   price_deviation_from_half — redundant with price_at_snapshot
#   total_volume           — redundant with log_volume
#   days_before_close      — redundant with target_percentile + duration_days
#   price_min/max_7d/14d   — captured by mean + volatility
#   price_range_7d/14d     — captured by volatility
#   price_mean_7d          — r=0.98 with price_at_snapshot, nearly redundant
#   price_mean_14d         — r=0.99 with price_mean_7d, r=0.96 with price_at_snapshot
#   pct_lifetime_elapsed   — r=0.93 with target_percentile; target_percentile is cleaner
#   outcome                — target variable
#   split                  — metadata
#   question               — raw text, needs NLP

print("\n" + "="*60)
print("SECTION 3: DEFINE FEATURES")
print("="*60)

numeric_features = [
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

categorical_features = ['category']
all_features         = numeric_features + categorical_features
target               = 'outcome'

print(f"Numeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features:       {len(all_features)}")

# Correlation check — confirm no highly correlated pairs remain
corr_matrix = train[numeric_features].corr().round(2)
high_corr = []
for i in range(len(numeric_features)):
    for j in range(i+1, len(numeric_features)):
        c = abs(corr_matrix.iloc[i, j])
        if c > 0.85:
            high_corr.append((numeric_features[i], numeric_features[j], c))

print(f"\nHighly correlated feature pairs (|r| > 0.85):")
if high_corr:
    for f1, f2, c in sorted(high_corr, key=lambda x: -x[2]):
        print(f"  {f1:30s} vs {f2:30s}  r={c:.2f}")
else:
    print("  None — feature set is clean")

# Check remaining correlations with price_at_snapshot specifically
print(f"\nCorrelations with price_at_snapshot:")
price_corrs = corr_matrix['price_at_snapshot'].drop('price_at_snapshot').sort_values(key=abs, ascending=False)
print(price_corrs.to_string())

print(f"\nMissing values per feature:")
print(train[all_features].isna().sum().to_string())

# ── 4. HANDLE RARE CATEGORIES AND PREPARE X/y ───────────────────────────────────
# We collapse rare category levels and prepare the feature matrix X and target
# vector y for both train and test. We also extract market_id groups for
# GroupKFold in section 6.
#
# With fixed percentile subsampling every market contributes exactly 7 rows,
# so category frequency should be computed at the MARKET level — not the row
# level — to avoid inflating frequent categories that happen to have more
# snapshots (which can't happen here by construction, but is still cleaner).

print("\n" + "="*60)
print("SECTION 4: HANDLE RARE CATEGORIES AND PREPARE X/y")
print("="*60)

RARE_THRESHOLD = 0.02

# Compute category frequency at market level
category_market_freq = (
    train.drop_duplicates('market_id')[['market_id', 'category']]
    .groupby('category')['market_id'].count()
    .div(train['market_id'].nunique())
)

rare_categories  = category_market_freq[category_market_freq < RARE_THRESHOLD].index.tolist()

print(f"Category frequencies (market level):")
print(category_market_freq.sort_values(ascending=False).round(3).to_string())
print(f"\nRare categories (< {RARE_THRESHOLD*100:.0f}% of markets): {rare_categories}")

# Collapse rare categories into 'other' in both train and test
train['category'] = train['category'].replace(rare_categories, 'other')
test['category']  = test['category'].replace(rare_categories, 'other')

# Handle categories in test unseen in train — map to 'other'
known_categories = set(train['category'].unique())
test['category']  = test['category'].apply(lambda x: x if x in known_categories else 'other')

print(f"\nFinal category levels: {sorted(train['category'].unique())}")
print(f"Test categories all known to train: {set(test['category'].unique()).issubset(known_categories)}")

# Outcome rate by category — confirms category is informative
print(f"\nOutcome rate by category:")
print(
    train.drop_duplicates('market_id')
    .groupby('category')['outcome']
    .mean()
    .sort_values(ascending=False)
    .round(3)
    .to_string()
)

# ── Prepare X and y ─────────────────────────────────────────────────────────────
train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

X_train = train[all_features]
y_train = train[target]
groups  = train['market_id']   # for GroupKFold in section 6

X_test  = test[all_features]
y_test  = test[target]

print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Unique markets in train: {groups.nunique():,}")

# ── 5. BUILD PREPROCESSING PIPELINE ────────────────────────────────────────────
# Identical structure to the original SVM implementation:
#   Numeric:      SimpleImputer (median) → StandardScaler
#   Categorical:  SimpleImputer (constant) → OneHotEncoder
#
# StandardScaler is critical for SVM — without it features with large ranges
# (e.g. duration_days: 30-972) dominate the kernel distance computation over
# features with small ranges (e.g. price_volatility_7d: 0-0.49).
#
# The preprocessor is NOT fit here on the full training set — it lives inside
# the Pipeline in section 6 so that during CV each fold fits the scaler and
# encoder only on its training portion, never on the validation fold.
# The fit_transform below is purely a sanity check on output shape and scaling.

print("\n" + "="*60)
print("SECTION 5: BUILD PREPROCESSING PIPELINE")
print("="*60)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ── Sanity check ─────────────────────────────────────────────────────────────────
preprocessor_check       = preprocessor.fit(X_train)
X_train_transformed      = preprocessor_check.transform(X_train)
X_test_transformed       = preprocessor_check.transform(X_test)

n_numeric  = len(numeric_features)
n_dummies  = len(train['category'].unique()) - 1  # drop='first' removes one level
n_expected = n_numeric + n_dummies

print(f"Expected transformed columns: {n_numeric} numeric + {n_dummies} dummies = {n_expected}")
print(f"Actual transformed shape — train: {X_train_transformed.shape}, test: {X_test_transformed.shape}")

# Confirm scaling: numeric columns should have mean ~0 and std ~1
transformed_means = X_train_transformed[:, :n_numeric].mean(axis=0).round(3)
transformed_stds  = X_train_transformed[:, :n_numeric].std(axis=0).round(3)
print(f"\nPost-scaling means (should be ~0): {transformed_means}")
print(f"Post-scaling stds  (should be ~1): {transformed_stds}")

# Confirm test set has same number of columns as train
assert X_train_transformed.shape[1] == X_test_transformed.shape[1], \
    "Train and test have different number of transformed columns"
print(f"\nTrain and test column counts match: ✓")

# ── 6. BUILD SVM PIPELINE AND CROSS-VALIDATION SETUP ───────────────────────────
# We chain the preprocessor and SVM into a single Pipeline object, then set up
# a single group-aware train/validation split for hyperparameter tuning.
#
# KEY DIFFERENCES FROM ORIGINAL SVM IMPLEMENTATION:
#   - No market subsampling needed — full 17,073 train markets is feasible
#     because fixed percentile subsampling reduced rows from 4M to 119k
#   - GroupKFold still used — even with 7 equal rows per market, snapshots
#     from the same market share the same outcome and correlated features,
#     so keeping them together in folds is still the cleaner approach
#   - class_weight='balanced' handles 82/18 class imbalance — no additional
#     snapshot count weighting needed since equal rows per market is already
#     guaranteed by construction
#
# SVC SETTINGS:
#   kernel='rbf'            — handles non-linear boundaries
#   probability=True        — enables predict_proba() via Platt scaling
#   class_weight='balanced' — upweights minority class (outcome=1)
#   C and gamma             — tuned in section 7 via GridSearchCV

print("\n" + "="*60)
print("SECTION 6: BUILD SVM PIPELINE AND CROSS-VALIDATION SETUP")
print("="*60)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, PredefinedSplit

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

print("Pipeline steps:")
for name, step in svm_pipeline.steps:
    print(f"  {name}: {step.__class__.__name__}")

print(f"\nSVM parameters (pre-tuning):")
print(f"  kernel:       {svm_pipeline['svm'].kernel}")
print(f"  probability:  {svm_pipeline['svm'].probability}")
print(f"  class_weight: {svm_pipeline['svm'].class_weight}")
print(f"  C:            {svm_pipeline['svm'].C}  ← will be tuned in section 7")
print(f"  gamma:        {svm_pipeline['svm'].gamma}  ← will be tuned in section 7")

# ── Single group-aware train/validation split ────────────────────────────────────
# We use a single 80/20 split rather than 5-fold CV to keep tuning feasible.
# GroupKFold ensures all 7 snapshots from a given market stay on the same
# side of the split — no market bleeds across train and validation.

N_SPLITS = 5
group_kfold = GroupKFold(n_splits=N_SPLITS)

# Use first fold as our single train/validation split
train_idx, val_idx = next(group_kfold.split(X_train, y_train, groups))

n_train_markets = groups.iloc[train_idx].nunique()
n_val_markets   = groups.iloc[val_idx].nunique()

# Confirm no market overlap
train_market_set = set(groups.iloc[train_idx])
val_market_set   = set(groups.iloc[val_idx])
overlap          = len(train_market_set & val_market_set)

# Build PredefinedSplit for GridSearchCV
split_indicator = np.full(len(X_train), -1)
split_indicator[val_idx] = 0
ps = PredefinedSplit(test_fold=split_indicator)

print(f"\nCross-validation strategy: single group-aware split")
print(f"  Train: {n_train_markets:,} markets ({len(train_idx):,} rows)")
print(f"  Val:   {n_val_markets:,} markets ({len(val_idx):,} rows)")
print(f"  Market overlap: {overlap}")  # must be 0

print("\n" + "="*60)
print("SECTION 7: HYPERPARAMETER TUNING")
print("="*60)

# ── 7. HYPERPARAMETER TUNING ───────────────────────────────────────────────────
# We use GridSearchCV with a coarse 3×4 grid over C and gamma.
# 12 combinations × 1 split = 12 total SVM fits.
#
# Based on findings from the original SVM implementation:
#   - Low gamma consistently outperformed high gamma → extended grid downward
#   - Best C varied between runs → keep full 0.1-10 range
#   - Single split is sufficient with 3,415 validation markets
#
# GRID:
#   C:     [0.1, 1.0, 10.0]        — underfitting to overfitting
#   gamma: [0.0001, 0.001, 0.01, 0.1] — global to local boundary

from sklearn.model_selection import GridSearchCV
import time

param_grid = {
    'svm__C':     [0.1, 1.0, 10.0],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1]
}

total_fits = len(param_grid['svm__C']) * len(param_grid['svm__gamma'])
print(f"Grid: {len(param_grid['svm__C'])} C values × "
      f"{len(param_grid['svm__gamma'])} gamma values = {total_fits} total fits")
print(f"C values:     {param_grid['svm__C']}")
print(f"Gamma values: {param_grid['svm__gamma']}")

search = GridSearchCV(
    svm_pipeline,
    param_grid=param_grid,
    cv=ps,              # single predefined group-aware split
    scoring='roc_auc',  # optimize for ROC-AUC
    n_jobs=-1,          # use all available CPU cores
    verbose=3,          # print progress
    refit=True          # refit best params on full X_train after search
)

print(f"\nStarting grid search ({total_fits} fits)...")
print(f"Training on {groups.nunique():,} markets ({len(X_train):,} rows)")

start   = time.time()
search.fit(X_train, y_train, groups=groups)
elapsed = time.time() - start

print(f"\nGrid search completed in {elapsed/60:.1f} minutes")
print(f"Best parameters: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ── Full results grid ────────────────────────────────────────────────────────────
cv_results = pd.DataFrame(search.cv_results_)

results_table = cv_results.pivot_table(
    index='param_svm__C',
    columns='param_svm__gamma',
    values='mean_test_score'
).round(4)

print(f"\nROC-AUC by C and gamma:")
print(results_table.to_string())

best_C     = search.best_params_['svm__C']
best_gamma = search.best_params_['svm__gamma']
print(f"\nBest: C={best_C}, gamma={best_gamma} → ROC-AUC={search.best_score_:.4f}")

# ── Save model ───────────────────────────────────────────────────────────────────
import joblib
model_path = ARTIFACTS_DIR / "svm_percentile_gridsearch.pkl"
joblib.dump(search, model_path)
print(f"\nModel saved to: {model_path}")
print(f"File size: {model_path.stat().st_size / (1024*1024):.1f} MB")