import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report


from scipy.stats import loguniform, uniform
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    filename='svm_run.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────────
# Read the parquet file. pyarrow is fast and memory-efficient for large files.
df = pd.read_parquet("../../../data/polymarket_ml_dataset.parquet", engine="fastparquet")

print(f"Full dataset: {df.shape}")
print(f"Unique markets: {df['market_id'].nunique()}")

# ── 2. BASIC CLEANING ───────────────────────────────────────────────────────────
# Parse timestamp (not used as a feature, but useful for diagnostics)
df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'], format='ISO8601', utc=True)

# Outcome must be integer 0/1 — confirm no NAs
assert df['outcome'].isna().sum() == 0, "NAs in outcome"
df['outcome'] = df['outcome'].astype(int)

# Fill NA categories with explicit "unknown" level rather than dropping rows
df['category'] = df['category'].fillna('unknown')
print(f"Unique categories: {df['category'].unique()}")

# ── 3. TRAIN/TEST SPLIT ─────────────────────────────────────────────────────────
# Your split is pre-defined at the market level — respect it.
# This means no market appears in both train and test, which prevents leakage.
train = df[df['split'] == 'train'].copy()
test  = df[df['split'] == 'test'].copy()

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train markets: {train['market_id'].nunique()}, Test markets: {test['market_id'].nunique()}")

# ── 4. SUBSAMPLE FOR SVM ────────────────────────────────────────────────────────
# SVM is O(n^2) to O(n^3) in training rows — 4M rows is not feasible on a laptop.
# We subsample at the MARKET level (not row level) to preserve group structure for CV.
#
# We use STRATIFIED sampling by category so the sample mirrors the full training
# set's category distribution. Pure random sampling risks dropping rare categories
# entirely, which would make OHE columns learned during training useless at test time.

N_MARKETS = 2000   # ~100k rows. Safe for local. Bump to 3000-5000 if running overnight.
RANDOM_STATE = 42

# One row per market with its category — this is what we stratify on
market_meta = (
    train
    .drop_duplicates('market_id')[['market_id', 'category']]
    .reset_index(drop=True)
)

# Stratified split: sample N_MARKETS markets, preserving category proportions
# StratifiedShuffleSplit expects a target array — we use category as the strata
sss = StratifiedShuffleSplit(n_splits=1, train_size=N_MARKETS, random_state=RANDOM_STATE)
sample_idx, _ = next(sss.split(market_meta, market_meta['category']))
sampled_markets = market_meta.iloc[sample_idx]['market_id'].values

train_sample = train[train['market_id'].isin(sampled_markets)].copy()

# ── Verify stratification held ──────────────────────────────────────────────────
cat_comparison = pd.DataFrame({
    'full_train': train.groupby('category')['market_id'].nunique() / train['market_id'].nunique(),
    'sample':     train_sample.groupby('category')['market_id'].nunique() / train_sample['market_id'].nunique()
}).round(3)

print(f"Sampled train: {train_sample.shape[0]:,} rows across {train_sample['market_id'].nunique():,} markets")
print(f"Avg snapshots per market: {train_sample.shape[0] / train_sample['market_id'].nunique():.1f}")
print(f"Outcome balance: {train_sample['outcome'].value_counts(normalize=True).round(3).to_dict()}")
print(f"\nCategory distribution (full train vs sample):")
print(cat_comparison.to_string())

# ── 5. DEFINE FEATURES ──────────────────────────────────────────────────────────
# We explicitly define which columns are features and what type they are.
# This controls what goes into the preprocessor in section 8.
#
# EXCLUDED columns and reasons:
#   market_id              — unique identifier, not generalizable to unseen markets
#   snapshot_timestamp     — raw time is meaningless; pct_lifetime_elapsed captures position in market life
#   price_deviation_from_half — mathematically redundant with price_at_snapshot (just |price - 0.5|)
#   total_volume           — redundant with log_volume; log scale is better for skewed data
#   days_before_close      — redundant with pct_lifetime_elapsed once we have duration_days
#   price_min_7d/14d       — captured by mean + range; adds multicollinearity
#   price_max_7d/14d       — same as above
#   price_range_7d/14d     — largely redundant with volatility
#   outcome                — target variable
#   split                  — metadata
#   question               — raw text; SVM requires numeric input, would need NLP embedding first

numeric_features = [
    'price_at_snapshot',      # crowd's probability estimate — will be strongest feature by far
    'pct_lifetime_elapsed',   # where we are in the market's life (0 = just opened, 1 = about to close)
    'duration_days',          # total market length — longer markets may behave differently
    'log_volume',             # liquidity proxy — high volume markets tend to be better calibrated
    'price_mean_7d',          # average price over past 7 days — smoothed recent signal
    'price_volatility_7d',    # price uncertainty over past 7 days — how settled is the market
    'price_change_7d',        # net price movement over past 7 days — recent momentum magnitude
    'price_trend_7d',         # slope of price over past 7 days — directional momentum
    'price_mean_14d',         # same as 7d equivalents but over longer window
    'price_volatility_14d',
    'price_change_14d',
    'price_trend_14d',
]

categorical_features = [
    'category'                # market topic — some categories may be systematically over/underpriced
]

all_features = numeric_features + categorical_features
target = 'outcome'

print(f"Numeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Total features:       {len(all_features)}")
print(f"\nNumeric feature summary:")
print(train_sample[numeric_features].describe().round(3).to_string())
print(f"\nMissing values per feature:")
print(train_sample[all_features].isna().sum().to_string())

# ── 6. HANDLE RARE CATEGORIES ───────────────────────────────────────────────────
# Before one-hot encoding, we collapse category levels that appear in fewer than
# 2% of MARKETS (not rows) into "other". We count at the market level because
# each market repeats across many snapshots — counting by row would artificially
# inflate the frequency of categories that happen to have longer markets.
#
# We also need to handle categories that appear in test but not in train — the
# OHE will have no column for them and will error. We map those to "other" too.

RARE_THRESHOLD = 0.02  # categories representing < 2% of markets get collapsed

# Compute frequency at the market level
category_market_freq = (
    train_sample
    .drop_duplicates('market_id')[['market_id', 'category']]
    .groupby('category')['market_id'].count()
    .div(train_sample['market_id'].nunique())
)

rare_categories = category_market_freq[category_market_freq < RARE_THRESHOLD].index.tolist()

print(f"Category frequencies (market level):")
print(category_market_freq.sort_values(ascending=False).round(3).to_string())
print(f"\nRare categories (< {RARE_THRESHOLD*100:.0f}% of markets): {rare_categories}")

# Collapse rare categories into "other" in both train and test
train_sample['category'] = train_sample['category'].replace(rare_categories, 'other')
test['category']         = test['category'].replace(rare_categories, 'other')

# Handle categories in test that never appeared in train at all
# (even after collapsing) — map them to "other"
known_categories = train_sample['category'].unique()
test['category'] = test['category'].apply(
    lambda x: x if x in known_categories else 'other'
)

print(f"\nFinal category levels after collapsing: {sorted(train_sample['category'].unique())}")
print(f"Test categories all known to train: {set(test['category'].unique()).issubset(set(known_categories))}")
print('Outcome distribution by category:')
print(train_sample.drop_duplicates('market_id').groupby('category')['outcome'].mean().sort_values(ascending=False).round(3))

# ── 7. PREPARE X AND y ──────────────────────────────────────────────────────────
# We extract the feature matrix X and target vector y for both train and test.
# We also extract the market_id groups from train — these are passed to GroupKFold
# in section 10 to ensure all snapshots from a given market stay in the same fold.
#
# We reset the index on all dataframes to avoid any index-alignment issues when
# sklearn internally slices arrays during CV and pipeline fitting.

train_sample = train_sample.reset_index(drop=True)
test         = test.reset_index(drop=True)

X_train = train_sample[all_features]
y_train = train_sample[target]
groups  = train_sample['market_id']   # used in section 10 for GroupKFold

X_test  = test[all_features]
y_test  = test[target]

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Groups (unique markets in train sample): {groups.nunique():,}")
print(f"\ny_train class balance:")
print(y_train.value_counts(normalize=True).round(3).to_string())
print(f"\ny_test class balance:")
print(y_test.value_counts(normalize=True).round(3).to_string())

# ── 8. BUILD PREPROCESSING PIPELINE ────────────────────────────────────────────
# ColumnTransformer applies different preprocessing to numeric and categorical
# columns simultaneously. Crucially, this lives inside the full Pipeline (built in
# section 9) so that when CV folds are created in section 10, the scaler and OHE
# are fit ONLY on the training fold and applied to the validation fold — never the
# reverse. This prevents data leakage through preprocessing.
#
# NUMERIC: StandardScaler → subtracts mean, divides by std → mean 0, std 1
#   SVM computes distances between points in feature space. Without scaling,
#   features with large ranges (e.g. duration_days: 30-971) dominate the distance
#   calculation over features with small ranges (e.g. price_volatility_7d: 0-0.49).
#   StandardScaler puts every feature on equal footing.
#
# CATEGORICAL: OneHotEncoder → converts each category level to a binary column
#   e.g. category='sports' → [1,0,0,0,0,0,0,0]
#        category='crypto' → [0,1,0,0,0,0,0,0]
#   drop='first' drops one column per feature to avoid perfect multicollinearity
#   (the dropped level becomes the reference — here that will be 'crypto'
#   alphabetically). handle_unknown='ignore' zeros out any category level seen
#   in test but not train — a safety net since we already handled this in section 6.
#
# IMPUTATION: no explicit imputer needed — section 5 confirmed zero missing values.
#   Adding one anyway as a safety net costs nothing and prevents silent failures
#   if the pipeline is reused on new data with missings.

from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # safety net for any future NAs
    ('scaler',  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # safety net
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Sanity check: fit on train, transform both — inspect output shape
# This confirms the preprocessor works before we attach the SVM to it
preprocessor_check = preprocessor.fit(X_train)
X_train_transformed = preprocessor_check.transform(X_train)
X_test_transformed  = preprocessor_check.transform(X_test)

# Expected columns: 12 numeric + 8 dummy columns (9 category levels - 1 dropped)
n_numeric = len(numeric_features)
n_dummies = len(train_sample['category'].unique()) - 1  # drop='first' removes one
print(f"Expected transformed columns: {n_numeric} numeric + {n_dummies} dummies = {n_numeric + n_dummies}")
print(f"Actual transformed shape — train: {X_train_transformed.shape}, test: {X_test_transformed.shape}")

# Confirm scaling worked: numeric columns should have mean ~0 and std ~1
transformed_means = X_train_transformed[:, :n_numeric].mean(axis=0).round(3)
transformed_stds  = X_train_transformed[:, :n_numeric].std(axis=0).round(3)
print(f"\nPost-scaling means (should be ~0): {transformed_means}")
print(f"Post-scaling stds  (should be ~1): {transformed_stds}")

# ── 9. BUILD SVM PIPELINE ───────────────────────────────────────────────────────
# A Pipeline chains preprocessing and model into a single object. This has two
# critical benefits:
#
#   1. LEAKAGE PREVENTION: during cross-validation in section 10, sklearn will
#      call pipeline.fit() on each training fold and pipeline.predict() on each
#      validation fold. Because the preprocessor is inside the pipeline, it gets
#      fit fresh on each training fold only — the validation fold is never seen
#      during fitting. If we preprocessed outside the pipeline, the scaler would
#      have already seen the entire training set including validation folds.
#
#   2. CONVENIENCE: at inference time, pipeline.predict(X_test) runs preprocessing
#      and prediction in one call — no need to manually transform before predicting.
#
# SVC SETTINGS:
#   kernel='rbf'          — radial basis function; handles non-linear boundaries.
#                           appropriate here since price/outcome relationship is
#                           unlikely to be linearly separable.
#   probability=True      — enables predict_proba() via Platt scaling (fits a
#                           logistic regression on top of SVM decision scores).
#                           required for Brier score evaluation and probability
#                           output. note: adds some training overhead.
#   class_weight='balanced' — upweights the minority class (outcome=1) during
#                           training. without this, the model is incentivized to
#                           just predict 0 for everything and achieve 83% accuracy.
#                           balanced sets weight = n_samples / (n_classes * n_samples_per_class)
#   C and gamma are left  — as tune() placeholders; RandomizedSearchCV in section
#                           11 will find the best values via cross-validation.

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svm', SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ))
])

# Confirm pipeline structure
print("Pipeline steps:")
for name, step in svm_pipeline.steps:
    print(f"  {name}: {step.__class__.__name__}")

print(f"\nSVM parameters (pre-tuning):")
print(f"  kernel:       {svm_pipeline['svm'].kernel}")
print(f"  probability:  {svm_pipeline['svm'].probability}")
print(f"  class_weight: {svm_pipeline['svm'].class_weight}")
print(f"  C:            {svm_pipeline['svm'].C}  ← will be tuned in section 11")
print(f"  gamma:        {svm_pipeline['svm'].gamma}  ← will be tuned in section 11")

# ── 10. CROSS-VALIDATION SETUP ──────────────────────────────────────────────────
# We use a single group-aware train/validation split rather than k-fold CV.
# This reduces total fits from n_combos × k to just n_combos × 1 — critical
# for making grid search feasible on a laptop.
#
# The group constraint is still enforced: all snapshots from a given market
# stay on the same side of the split. We split at the MARKET level (80/20),
# then derive row-level assignments from market membership.
#
# PredefinedSplit is sklearn's way of specifying a fixed split:
#   -1 = this row is in the training portion
#    0 = this row is in the validation portion

from sklearn.model_selection import PredefinedSplit

# Split markets 80/20 — sampled_markets was randomly ordered in section 4
# so this is effectively a random market-level split
n_val_markets    = int(len(sampled_markets) * 0.2)
train_market_set = set(sampled_markets[:-n_val_markets])
val_market_set   = set(sampled_markets[-n_val_markets:])

# Assign each ROW a split label based on which market it belongs to
split_indicator = np.where(
    train_sample['market_id'].isin(val_market_set), 0, -1
)

ps = PredefinedSplit(test_fold=split_indicator)

# Confirm split integrity
n_train_rows = (split_indicator == -1).sum()
n_val_rows   = (split_indicator ==  0).sum()
overlap      = len(train_market_set & val_market_set)

print(f"Cross-validation strategy: single predefined split (group-aware)")
print(f"  Train: {len(train_market_set):,} markets ({n_train_rows:,} rows)")
print(f"  Val:   {len(val_market_set):,} markets ({n_val_rows:,} rows)")
print(f"  Market overlap: {overlap}")  # must be 0

# ── 11. HYPERPARAMETER TUNING ───────────────────────────────────────────────────
# We use GridSearchCV with a coarse 3×3 grid over C and gamma.
# 9 combinations × 1 split = 9 total SVM fits — feasible on a laptop.
#
# WHY COARSE GRID OVER RANDOM SEARCH:
#   With only a handful of combinations to try, a structured grid is more reliable
#   than random sampling — it guarantees coverage of low/medium/high values for
#   both hyperparameters rather than risking clustering in one region by chance.
#   It also produces a readable C × gamma results table showing how performance
#   varies across the grid, which is useful for interpretation and writeup.
#
# GRID:
#   C:     [0.1, 1.0, 10.0]   — covers underfitting (low C) to overfitting (high C)
#   gamma: [0.001, 0.01, 0.1] — covers global (low gamma) to local (high gamma)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C':     [0.1, 1.0, 10.0],
    'svm__gamma': [0.0001, 0.001, 0.01, 0.1]
}

total_fits = len(param_grid['svm__C']) * len(param_grid['svm__gamma'])
print(f"\nGrid: {len(param_grid['svm__C'])} C values × "
      f"{len(param_grid['svm__gamma'])} gamma values = {total_fits} total fits")
print(f"C values:     {param_grid['svm__C']}")
print(f"Gamma values: {param_grid['svm__gamma']}")

search = GridSearchCV(
    svm_pipeline,
    param_grid=param_grid,
    cv=ps,              # single predefined group-aware split
    scoring='roc_auc',  # optimize for ROC-AUC
    n_jobs=-1,          # use all CPU cores
    verbose=3,          # print progress so you know it's running
    refit=True          # refit best params on full train_sample after search
)

print(f"\nStarting grid search ({total_fits} fits)...")
print(f"Running on {train_sample['market_id'].nunique():,} markets — "
      f"{'QUICK TEST RUN' if train_sample['market_id'].nunique() <= 200 else 'FULL RUN'}")

import time
start   = time.time()
search.fit(X_train, y_train, groups=groups)
elapsed = time.time() - start

print(f"\nGrid search completed in {elapsed/60:.1f} minutes")
print(f"Best parameters: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ── Full results grid ───────────────────────────────────────────────────────────
# Pivot into a readable C × gamma table so you can see performance trends clearly
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

# ── 12. EVALUATE ON TEST SET ────────────────────────────────────────────────────
# We evaluate the final model on the held-out test set — markets the model has
# never seen at any point during training or tuning.
#
# Two metrics:
#   ROC-AUC:     measures the model's ability to RANK positive cases above negative
#                cases. Invariant to class imbalance — a random classifier scores
#                0.5, a perfect classifier scores 1.0.
#
#   Brier score: measures PROBABILITY CALIBRATION — how close predicted
#                probabilities are to true outcomes. Defined as mean((p - y)^2).
#                Lower is better. A model predicting the base rate (0.162) for
#                every row scores 0.162 × (1 - 0.162) ≈ 0.136. A perfect model
#                scores 0.0. This is the more meaningful metric for prediction
#                markets where you care about probability estimates, not just ranking.
#
# We also compare against a naive baseline: just use price_at_snapshot directly
# as the probability estimate. This is the crowd's own prediction — if our model
# can't beat it, the additional features add no value.

best_model = search.best_estimator_

# Get predicted probabilities — we want P(outcome=1), which is the second column
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_class = best_model.predict(X_test)

# ── Overall metrics ─────────────────────────────────────────────────────────────
roc_auc = roc_auc_score(y_test, y_pred_proba)
brier   = brier_score_loss(y_test, y_pred_proba)

# Baseline: use price_at_snapshot directly as probability
baseline_roc_auc = roc_auc_score(y_test, test['price_at_snapshot'])
baseline_brier   = brier_score_loss(y_test, test['price_at_snapshot'])

print(f"── Overall Test Set Performance ──────────────────────────")
print(f"{'Metric':<20} {'SVM':>10} {'Baseline':>10} {'Improvement':>12}")
print(f"{'-'*54}")
print(f"{'ROC-AUC':<20} {roc_auc:>10.4f} {baseline_roc_auc:>10.4f} {roc_auc - baseline_roc_auc:>+12.4f}")
print(f"{'Brier Score':<20} {brier:>10.4f} {baseline_brier:>10.4f} {baseline_brier - brier:>+12.4f}")
print(f"\n{classification_report(y_test, y_pred_class, target_names=['outcome=0', 'outcome=1'])}")

# ── Support vector summary ───────────────────────────────────────────────────────
# A very high number of support vectors relative to training size suggests
# the boundary is complex and the model may be overfitting
n_support = best_model.named_steps['svm'].n_support_
print(f"── Support Vectors ───────────────────────────────────────")
print(f"  Class 0: {n_support[0]:,}")
print(f"  Class 1: {n_support[1]:,}")
print(f"  Total:   {sum(n_support):,} ({sum(n_support)/len(X_train)*100:.1f}% of training rows)")

# ── Stratified evaluation by lifetime stage ─────────────────────────────────────
# Overall metrics are dominated by mid/late snapshots where prediction is easier.
# Breaking down by lifetime stage reveals where the model genuinely adds signal
# vs. just calling already-converged prices correctly.

test_eval = test.copy()
test_eval['pred_proba'] = y_pred_proba
test_eval['pred_class'] = y_pred_class
test_eval['lifetime_bin'] = pd.cut(
    test_eval['pct_lifetime_elapsed'],
    bins=[0, 0.33, 0.67, 1.0],
    labels=['early (0-33%)', 'mid (33-67%)', 'late (67-100%)']
)

print(f"\n── Performance by Lifetime Stage ─────────────────────────")
print(f"{'Stage':<20} {'ROC-AUC':>10} {'Brier':>10} {'Markets':>10} {'Rows':>10}")
print(f"{'-'*62}")
for stage in ['early (0-33%)', 'mid (33-67%)', 'late (67-100%)']:
    subset = test_eval[test_eval['lifetime_bin'] == stage]
    if len(subset) == 0:
        continue
    stage_auc    = roc_auc_score(subset['outcome'], subset['pred_proba'])
    stage_brier  = brier_score_loss(subset['outcome'], subset['pred_proba'])
    n_mkts       = subset['market_id'].nunique()
    print(f"{stage:<20} {stage_auc:>10.4f} {stage_brier:>10.4f} {n_mkts:>10,} {len(subset):>10,}")

# ── Baseline comparison by lifetime stage ───────────────────────────────────────
print(f"\n── Baseline (price_at_snapshot) by Lifetime Stage ────────")
print(f"{'Stage':<20} {'ROC-AUC':>10} {'Brier':>10}")
print(f"{'-'*42}")
for stage in ['early (0-33%)', 'mid (33-67%)', 'late (67-100%)']:
    subset = test_eval[test_eval['lifetime_bin'] == stage]
    if len(subset) == 0:
        continue
    stage_auc   = roc_auc_score(subset['outcome'], subset['price_at_snapshot'])
    stage_brier = brier_score_loss(subset['outcome'], subset['price_at_snapshot'])
    print(f"{stage:<20} {stage_auc:>10.4f} {stage_brier:>10.4f}")