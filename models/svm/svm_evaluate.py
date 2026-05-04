import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
RANDOM_STATE  = 42

# ── LOAD SUBSAMPLED DATA ─────────────────────────────────────────────────────────
df_sampled = pd.read_parquet(ARTIFACTS_DIR / "df_sampled.parquet")
df_sampled['category'] = df_sampled['category'].fillna('unknown')
df_sampled['outcome']  = df_sampled['outcome'].astype(int)

train = df_sampled[df_sampled['split'] == 'train'].copy().reset_index(drop=True)
test  = df_sampled[df_sampled['split'] == 'test'].copy().reset_index(drop=True)

# ── LOAD SAVED MODEL ─────────────────────────────────────────────────────────────
search     = joblib.load(ARTIFACTS_DIR / "svm_percentile_gridsearch.pkl")
best_model = search.best_estimator_

print(f"Loaded model — best params: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ── FEATURE DEFINITIONS ──────────────────────────────────────────────────────────
numeric_features = [
    'price_at_snapshot', 'target_percentile', 'duration_days', 'log_volume',
    'price_volatility_7d', 'price_change_7d', 'price_trend_7d',
    'price_volatility_14d', 'price_change_14d', 'price_trend_14d',
]
categorical_features = ['category']
all_features         = numeric_features + categorical_features
target               = 'outcome'

# ── RECREATE CATEGORY CLEANING ───────────────────────────────────────────────────
RARE_THRESHOLD = 0.02
category_market_freq = (
    train.drop_duplicates('market_id')[['market_id', 'category']]
    .groupby('category')['market_id'].count()
    .div(train['market_id'].nunique())
)
rare_categories  = category_market_freq[category_market_freq < RARE_THRESHOLD].index.tolist()
train['category'] = train['category'].replace(rare_categories, 'other')
test['category']  = test['category'].replace(rare_categories, 'other')
known_categories  = set(train['category'].unique())
test['category']  = test['category'].apply(lambda x: x if x in known_categories else 'other')

# ── PREPARE X AND y ──────────────────────────────────────────────────────────────
X_train = train[all_features]
y_train = train[target]
X_test  = test[all_features]
y_test  = test[target]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── SECTION 8: EVALUATE ON TEST SET ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8: EVALUATE ON TEST SET")
print("="*60)

# Generate predictions
# predict_proba returns [P(outcome=0), P(outcome=1)] — we want the second column
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred_class = best_model.predict(X_test)

# ── Overall metrics ──────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

roc_auc = roc_auc_score(y_test, y_pred_proba)
brier   = brier_score_loss(y_test, y_pred_proba)
ap      = average_precision_score(y_test, y_pred_proba)

# Baseline: use price_at_snapshot directly as probability
baseline_proba   = test['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
baseline_auc     = roc_auc_score(y_test, baseline_proba)
baseline_brier   = brier_score_loss(y_test, baseline_proba)
baseline_ap      = average_precision_score(y_test, baseline_proba)

print(f"{'Metric':<25} {'SVM':>10} {'Baseline':>10} {'Improvement':>12}")
print(f"{'-'*59}")
print(f"{'ROC-AUC':<25} {roc_auc:>10.4f} {baseline_auc:>10.4f} {roc_auc - baseline_auc:>+12.4f}")
print(f"{'Brier Score':<25} {brier:>10.4f} {baseline_brier:>10.4f} {baseline_brier - brier:>+12.4f}")
print(f"{'Avg Precision':<25} {ap:>10.4f} {baseline_ap:>10.4f} {ap - baseline_ap:>+12.4f}")
print(f"\n{classification_report(y_test, y_pred_class, target_names=['outcome=0', 'outcome=1'])}")

# ── Support vector summary ───────────────────────────────────────────────────────
n_support = best_model.named_steps['svm'].n_support_
print(f"Support Vectors:")
print(f"  Class 0: {n_support[0]:,}")
print(f"  Class 1: {n_support[1]:,}")
print(f"  Total:   {sum(n_support):,} ({sum(n_support)/len(X_train)*100:.1f}% of training rows)")

# ── Market-level metrics (corrects for snapshot count bias) ──────────────────────
# Since every market contributes exactly 7 rows, row-weighted and market-weighted
# metrics should be very similar — but we report both for completeness
test_eval = test.copy()
test_eval['pred_proba'] = y_pred_proba
test_eval['pred_class'] = y_pred_class
test_eval['correct']    = (test_eval['pred_class'] == test_eval['outcome']).astype(int)

market_metrics = test_eval.groupby('market_id').agg(
    accuracy    = ('correct', 'mean'),
    brier       = ('pred_proba', lambda x: np.mean(
                      (x - test_eval.loc[x.index, 'outcome'])**2)),
    outcome     = ('outcome', 'first'),
    n_snapshots = ('market_id', 'size')
).reset_index()

print(f"\nRow-weighted    — Accuracy: {test_eval['correct'].mean():.4f}  Brier: {brier:.4f}")
print(f"Market-weighted — Accuracy: {market_metrics['accuracy'].mean():.4f}  Brier: {market_metrics['brier'].mean():.4f}")

# ── Stratified evaluation by lifetime stage ──────────────────────────────────────
# With fixed percentile subsampling, target_percentile IS the lifetime stage —
# we can evaluate performance at each of the 7 standardized snapshot points.
# This is more precise than binning pct_lifetime_elapsed into broad buckets.

print(f"\n── Performance by Lifetime Stage ─────────────────────────────────────────")
print(f"{'Percentile':<15} {'ROC-AUC':>10} {'Brier':>10} {'Baseline AUC':>14} {'Baseline Brier':>15} {'Markets':>10}")
print(f"{'-'*76}")

for pct in sorted(test_eval['target_percentile'].unique()):
    subset      = test_eval[test_eval['target_percentile'] == pct]
    sub_base    = subset['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
    stage_auc   = roc_auc_score(subset['outcome'], subset['pred_proba'])
    stage_brier = brier_score_loss(subset['outcome'], subset['pred_proba'])
    base_auc    = roc_auc_score(subset['outcome'], sub_base)
    base_brier  = brier_score_loss(subset['outcome'], sub_base)
    n_mkts      = subset['market_id'].nunique()
    print(f"{pct:<15.2f} {stage_auc:>10.4f} {stage_brier:>10.4f} {base_auc:>14.4f} {base_brier:>15.4f} {n_mkts:>10,}")

# ── Performance by category ──────────────────────────────────────────────────────
print(f"\n── Performance by Category ────────────────────────────────────────────────")
print(f"{'Category':<20} {'ROC-AUC':>10} {'Brier':>10} {'Baseline AUC':>14} {'Outcome Rate':>14} {'Markets':>10}")
print(f"{'-'*80}")

for cat in sorted(test_eval['category'].unique()):
    subset      = test_eval[test_eval['category'] == cat]
    sub_base    = subset['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
    try:
        cat_auc = roc_auc_score(subset['outcome'], subset['pred_proba'])
        base_auc = roc_auc_score(subset['outcome'], sub_base)
    except ValueError:
        cat_auc = base_auc = float('nan')  # only one class in subset
    cat_brier    = brier_score_loss(subset['outcome'], subset['pred_proba'])
    outcome_rate = subset['outcome'].mean()
    n_mkts       = subset['market_id'].nunique()
    print(f"{cat:<20} {cat_auc:>10.4f} {cat_brier:>10.4f} {base_auc:>14.4f} {outcome_rate:>14.3f} {n_mkts:>10,}")

# ── SECTION 9: PLOTS ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 9: PLOTS")
print("="*60)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

# Plot styling
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

PLOT_DIR = ARTIFACTS_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Plot 1: ROC Curve ────────────────────────────────────────────────────────────
print("Generating Plot 1: ROC Curve...")
fpr_svm,  tpr_svm,  _ = roc_curve(y_test, y_pred_proba)
fpr_base, tpr_base, _ = roc_curve(y_test, baseline_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_svm,  tpr_svm,  label=f'SVM (AUC={roc_auc:.4f})',          color='steelblue', lw=2)
ax.plot(fpr_base, tpr_base, label=f'Baseline (AUC={baseline_auc:.4f})', color='coral',     lw=2, linestyle='--')
ax.plot([0, 1], [0, 1],     label='Random classifier (AUC=0.50)',        color='gray',      lw=1, linestyle=':')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — SVM vs Baseline')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'roc_curve.png', dpi=150)
plt.close()

# ── Plot 2: Precision-Recall Curve ───────────────────────────────────────────────
print("Generating Plot 2: Precision-Recall Curve...")
prec_svm,  rec_svm,  _ = precision_recall_curve(y_test, y_pred_proba)
prec_base, rec_base, _ = precision_recall_curve(y_test, baseline_proba)
baseline_rate           = y_test.mean()

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_svm,  prec_svm,  label=f'SVM (AP={ap:.4f})',               color='steelblue', lw=2)
ax.plot(rec_base, prec_base, label=f'Baseline (AP={baseline_ap:.4f})',  color='coral',     lw=2, linestyle='--')
ax.axhline(baseline_rate, color='gray', lw=1, linestyle=':', label=f'No-skill baseline ({baseline_rate:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — SVM vs Baseline')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'precision_recall_curve.png', dpi=150)
plt.close()

# ── Plot 3: Calibration Plot ─────────────────────────────────────────────────────
print("Generating Plot 3: Calibration Plot...")
prob_true_svm,  prob_pred_svm  = calibration_curve(y_test, y_pred_proba, n_bins=20)
prob_true_base, prob_pred_base = calibration_curve(y_test, baseline_proba, n_bins=20)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred_svm,  prob_true_svm,  label='SVM',      color='steelblue', lw=2, marker='o', markersize=4)
ax.plot(prob_pred_base, prob_true_base, label='Baseline',  color='coral',     lw=2, marker='s', markersize=4, linestyle='--')
ax.plot([0, 1], [0, 1], label='Perfect calibration',       color='gray',      lw=1, linestyle=':')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Plot — SVM vs Baseline')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'calibration.png', dpi=150)
plt.close()

# ── Plot 4: ROC-AUC by Lifetime Stage ───────────────────────────────────────────
print("Generating Plot 4: ROC-AUC by Lifetime Stage...")
percentiles  = sorted(test_eval['target_percentile'].unique())
svm_aucs     = []
base_aucs    = []

for pct in percentiles:
    subset   = test_eval[test_eval['target_percentile'] == pct]
    sub_base = subset['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
    svm_aucs.append(roc_auc_score(subset['outcome'], subset['pred_proba']))
    base_aucs.append(roc_auc_score(subset['outcome'], sub_base))

x     = np.arange(len(percentiles))
width = 0.35
pct_labels = [f'{int(p*100)}%' for p in percentiles]

fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width/2, svm_aucs,  width, label='SVM',      color='steelblue', alpha=0.85)
ax.bar(x + width/2, base_aucs, width, label='Baseline',  color='coral',     alpha=0.85)
ax.set_xlabel('Lifetime Percentile')
ax.set_ylabel('ROC-AUC')
ax.set_title('ROC-AUC by Lifetime Stage — SVM vs Baseline')
ax.set_xticks(x)
ax.set_xticklabels(pct_labels)
ax.set_ylim(0.88, 0.98)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / 'auc_by_lifetime_stage.png', dpi=150)
plt.close()

# ── Plot 5: Brier Score by Lifetime Stage ────────────────────────────────────────
print("Generating Plot 5: Brier Score by Lifetime Stage...")
svm_briers  = []
base_briers = []

for pct in percentiles:
    subset   = test_eval[test_eval['target_percentile'] == pct]
    sub_base = subset['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
    svm_briers.append(brier_score_loss(subset['outcome'], subset['pred_proba']))
    base_briers.append(brier_score_loss(subset['outcome'], sub_base))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pct_labels, svm_briers,  label='SVM',     color='steelblue', lw=2, marker='o')
ax.plot(pct_labels, base_briers, label='Baseline', color='coral',     lw=2, marker='s', linestyle='--')
ax.set_xlabel('Lifetime Percentile')
ax.set_ylabel('Brier Score (lower is better)')
ax.set_title('Brier Score by Lifetime Stage — SVM vs Baseline')
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / 'brier_by_lifetime_stage.png', dpi=150)
plt.close()

# ── Plot 6: ROC-AUC by Category ─────────────────────────────────────────────────
print("Generating Plot 6: ROC-AUC by Category...")
categories    = sorted(test_eval['category'].unique())
svm_cat_aucs  = []
base_cat_aucs = []

for cat in categories:
    subset   = test_eval[test_eval['category'] == cat]
    sub_base = subset['price_at_snapshot'].clip(1e-6, 1 - 1e-6)
    try:
        svm_cat_aucs.append(roc_auc_score(subset['outcome'], subset['pred_proba']))
        base_cat_aucs.append(roc_auc_score(subset['outcome'], sub_base))
    except ValueError:
        svm_cat_aucs.append(np.nan)
        base_cat_aucs.append(np.nan)

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, svm_cat_aucs,  width, label='SVM',     color='steelblue', alpha=0.85)
ax.bar(x + width/2, base_cat_aucs, width, label='Baseline', color='coral',     alpha=0.85)
ax.set_xlabel('Category')
ax.set_ylabel('ROC-AUC')
ax.set_title('ROC-AUC by Category — SVM vs Baseline')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=30, ha='right')
ax.set_ylim(0.78, 0.98)
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / 'auc_by_category.png', dpi=150)
plt.close()

# ── Plot 7: Predicted Probability Distribution ───────────────────────────────────
print("Generating Plot 7: Predicted Probability Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (preds, title) in zip(axes, [
    (y_pred_proba, 'SVM Predicted Probabilities'),
    (baseline_proba, 'Baseline (price_at_snapshot)')
]):
    ax.hist(preds[y_test == 0], bins=50, alpha=0.6, label='outcome=0', color='coral',     density=True)
    ax.hist(preds[y_test == 1], bins=50, alpha=0.6, label='outcome=1', color='steelblue', density=True)
    ax.set_xlabel('Predicted P(outcome=1)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

plt.suptitle('Predicted Probability Distributions by Outcome', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'probability_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 8: Prediction Error by Lifetime Stage ───────────────────────────────────
print("Generating Plot 8: Prediction Error by Lifetime Stage...")
test_eval['error']     = test_eval['pred_proba'] - test_eval['outcome']
test_eval['abs_error'] = test_eval['error'].abs()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Error distribution by percentile — boxplot
test_eval.boxplot(
    column='error',
    by='target_percentile',
    ax=axes[0],
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', alpha=0.6)
)
axes[0].axhline(0, color='red', lw=1, linestyle='--')
axes[0].set_xlabel('Lifetime Percentile')
axes[0].set_ylabel('Prediction Error (pred - actual)')
axes[0].set_title('Prediction Error Distribution by Lifetime Stage')
axes[0].set_xticklabels(pct_labels)

# Mean absolute error by percentile — bar chart
mae_by_pct = test_eval.groupby('target_percentile')['abs_error'].mean()
axes[1].bar(pct_labels, mae_by_pct.values, color='steelblue', alpha=0.85)
axes[1].set_xlabel('Lifetime Percentile')
axes[1].set_ylabel('Mean Absolute Error')
axes[1].set_title('Mean Absolute Error by Lifetime Stage')

plt.suptitle('Prediction Errors by Lifetime Stage', fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'error_by_lifetime_stage.png', dpi=150)
plt.close()

print(f"\nAll plots saved to: {PLOT_DIR}")
print(f"Plots generated:")
for f in sorted(PLOT_DIR.glob('*.png')):
    print(f"  {f.name}")