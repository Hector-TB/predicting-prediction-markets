"""
Data Exploration & Validation
================================
Validates polymarket_markets_meta.csv and polymarket_ml_dataset.csv,
surfaces data quality issues, and saves diagnostic plots to plots/.

Run after build_snapshots.py (and optionally categorize_markets.py):
    python explore_data.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
PLOTS_DIR   = ROOT / "plots"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

META_CSV    = DATA_DIR / "polymarket_markets_meta.csv"
DATASET_CSV = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
SEP         = "=" * 60

ROLLING_7D  = [
    "price_mean_7d", "price_volatility_7d", "price_min_7d",
    "price_max_7d",  "price_change_7d",     "price_range_7d", "price_trend_7d",
]
ROLLING_14D = [c.replace("_7d", "_14d") for c in ROLLING_7D]

sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def save_fig(name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# META CSV
# ─────────────────────────────────────────────

def explore_meta(meta: pd.DataFrame) -> int:
    """Validate markets meta CSV. Returns duplicate market_id count."""
    section("MARKETS META")

    print(f"\nShape: {meta.shape[0]:,} rows x {meta.shape[1]} columns")
    print(f"Columns: {list(meta.columns)}")

    # Duplicates
    print(f"\n--- Duplicates ---")
    dupe_ids = meta["market_id"].duplicated().sum()
    print(f"  Duplicate market_ids: {dupe_ids}")
    if dupe_ids > 0:
        print(f"  ACTION NEEDED: {dupe_ids} duplicate market_ids — deduplicate before modelling")

    # Missing values
    print(f"\n--- Missing Values ---")
    nulls = meta.isnull().sum()
    if nulls.any():
        print(nulls[nulls > 0].to_string())
    else:
        print("  None")
    cat_missing_pct = meta["category"].isna().mean()
    print(f"  NOTE: category is {cat_missing_pct:.1%} missing — run categorize_markets.py first")

    # Outcome distribution
    print(f"\n--- Outcome Distribution ---")
    vc = meta["outcome"].value_counts()
    for val, count in vc.items():
        print(f"  {val} ({'YES' if val == 1 else 'NO '}): {count:>6,}  ({count / len(meta):.1%})")

    # Duration
    print(f"\n--- Duration (days) ---")
    d = meta["duration_days"]
    print(f"  min={d.min()}  median={d.median():.0f}  mean={d.mean():.0f}  max={d.max()}")
    print(f"  30-60d:   {((d >= 30) & (d < 60)).sum():>6,}")
    print(f"  60-90d:   {((d >= 60) & (d < 90)).sum():>6,}")
    print(f"  90-180d:  {((d >= 90) & (d < 180)).sum():>6,}")
    print(f"  180-365d: {((d >= 180) & (d < 365)).sum():>6,}")
    print(f"  365d+:    {(d >= 365).sum():>6,}")

    # Date range
    print(f"\n--- Date Range ---")
    print(f"  Earliest start: {meta['start_date'].min().date()}")
    print(f"  Latest start:   {meta['start_date'].max().date()}")
    print(f"  Earliest end:   {meta['end_date'].min().date()}")
    print(f"  Latest end:     {meta['end_date'].max().date()}")

    # Volume
    print(f"\n--- Volume (USD) ---")
    v = meta["total_volume"]
    print(f"  min=    ${v.min():>12,.0f}")
    print(f"  median= ${v.median():>12,.0f}")
    print(f"  mean=   ${v.mean():>12,.0f}")
    print(f"  max=    ${v.max():>12,.0f}")
    print(f"  $1k-$10k:     {((v >= 1_000) & (v < 10_000)).sum():>6,}")
    print(f"  $10k-$100k:   {((v >= 10_000) & (v < 100_000)).sum():>6,}")
    print(f"  $100k-$1M:    {((v >= 100_000) & (v < 1_000_000)).sum():>6,}")
    print(f"  $1M+:         {(v >= 1_000_000).sum():>6,}")

    # yes_final_price sanity
    print(f"\n--- Yes Final Price Sanity ---")
    yp       = meta["yes_final_price"]
    near_one  = (yp >= 0.95).sum()
    near_zero = (yp <= 0.05).sum()
    mid       = ((yp > 0.05) & (yp < 0.95)).sum()
    print(f"  >= 0.95 (YES): {near_one:>6,}  ({near_one / len(meta):.1%})")
    print(f"  <= 0.05 (NO):  {near_zero:>6,}  ({near_zero / len(meta):.1%})")
    print(f"  0.05-0.95:     {mid:>6,}  <- should be 0 (ambiguous filtered out)")

    # Train/test split
    print(f"\n--- Train/Test Split ---")
    sp = meta["split"].value_counts()
    for split, count in sp.items():
        print(f"  {split}: {count:>6,}  ({count / len(meta):.1%})")
    for split in ["train", "test"]:
        sub     = meta[meta["split"] == split]
        yes_pct = sub["outcome"].mean()
        print(f"  {split} outcome balance: YES={yes_pct:.1%} NO={1 - yes_pct:.1%}")

    # ── Plots ──────────────────────────────────

    # Duration histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(meta["duration_days"], bins=50, color="#4CAF50", edgecolor="white")
    ax.axvline(meta["duration_days"].median(), color="red", linestyle="--",
               label=f"Median = {meta['duration_days'].median():.0f}d")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Number of Markets")
    ax.set_title("Market Duration Distribution")
    ax.legend()
    plt.tight_layout()
    save_fig("meta_duration_hist.png")

    # Volume histogram (log scale)
    log_vol = np.log10(meta["total_volume"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_vol, bins=50, color="#FF9800", edgecolor="white")
    ax.axvline(np.log10(meta["total_volume"].median()), color="red", linestyle="--",
               label=f"Median = ${meta['total_volume'].median():,.0f}")
    ticks = [3, 4, 5, 6, 7, 8, 9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{t}}}$" for t in ticks])
    ax.set_xlabel("Total Volume (USD, log scale)")
    ax.set_ylabel("Number of Markets")
    ax.set_title("Total Volume Distribution")
    ax.legend()
    plt.tight_layout()
    save_fig("meta_volume_hist.png")

    return int(dupe_ids)


# ─────────────────────────────────────────────
# SNAPSHOT DATASET
# ─────────────────────────────────────────────

def explore_dataset(df: pd.DataFrame, meta_dupe_count: int = 0) -> None:
    """Validate snapshot dataset CSV."""
    section("SNAPSHOT DATASET")

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Duplicates
    print(f"\n--- Duplicates ---")
    dupe_snaps = df.duplicated(subset=["market_id", "snapshot_timestamp"]).sum()
    print(f"  Duplicate (market_id, snapshot_timestamp): {dupe_snaps}")

    # Missing values
    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    if nulls.any():
        print(nulls[nulls > 0].to_string())
    else:
        print("  None")

    # NaN rolling feature breakdown
    total_markets = df["market_id"].nunique()
    nan_7d_mask    = df[ROLLING_7D].isna().any(axis=1)
    nan_7d_markets = df[nan_7d_mask]["market_id"].nunique()
    nan_14d_mask    = df[ROLLING_14D].isna().any(axis=1)
    nan_14d_markets = df[nan_14d_mask]["market_id"].nunique()
    print(f"\n--- NaN Rolling Feature Breakdown ---")
    print(f"  7d  NaN rows:    {nan_7d_mask.sum():>9,} ({nan_7d_mask.mean():.1%} of snapshots)")
    print(f"  7d  NaN markets: {nan_7d_markets:>9,} ({nan_7d_markets / total_markets:.1%} of markets)")
    print(f"  14d NaN rows:    {nan_14d_mask.sum():>9,} ({nan_14d_mask.mean():.1%} of snapshots)")
    print(f"  14d NaN markets: {nan_14d_markets:>9,} ({nan_14d_markets / total_markets:.1%} of markets)")
    print(f"  NOTE: NaNs occur at early snapshots where < 7 or 14 days of history exist (expected)")

    # Market coverage
    print(f"\n--- Market Coverage ---")
    n_markets = total_markets
    snaps_per = df.groupby("market_id").size()
    print(f"  Unique markets:          {n_markets:>7,}")
    print(f"  Total snapshots:         {len(df):>7,}")
    print(f"  Snapshots/market min:    {snaps_per.min():>7,}")
    print(f"  Snapshots/market median: {snaps_per.median():>7.0f}")
    print(f"  Snapshots/market mean:   {snaps_per.mean():>7.1f}")
    print(f"  Snapshots/market max:    {snaps_per.max():>7,}")

    # Outcome distribution
    print(f"\n--- Outcome Distribution (snapshot level) ---")
    vc = df["outcome"].value_counts()
    for val, count in vc.items():
        print(f"  {val} ({'YES' if val == 1 else 'NO '}): {count:>8,}  ({count / len(df):.1%})")

    # Time features
    print(f"\n--- Time Features ---")
    dbc = df["days_before_close"]
    print(f"  days_before_close:")
    print(f"    min={dbc.min():.1f}  median={dbc.median():.1f}  max={dbc.max():.1f}")
    print(f"    < 14 days: {(dbc < 14).sum():>7,}  <- should be 0 (cutoff enforced)")
    ple = df["pct_lifetime_elapsed"]
    print(f"  pct_lifetime_elapsed:")
    print(f"    min={ple.min():.3f}  median={ple.median():.3f}  max={ple.max():.3f}")
    print(f"    > 1.0: {(ple > 1.0).sum()}  <- should be 0")
    print(f"    < 0.0: {(ple < 0.0).sum()}  <- should be 0")

    # Price features
    print(f"\n--- Price at Snapshot ---")
    p = df["price_at_snapshot"]
    print(f"  min={p.min():.4f}  median={p.median():.4f}  mean={p.mean():.4f}  max={p.max():.4f}")
    out_of_range = ((p < 0) | (p > 1)).sum()
    print(f"  Out of [0,1]: {out_of_range}  <- should be 0")
    if out_of_range > 0:
        bad = df[df["price_at_snapshot"] > 1.0][["market_id", "price_at_snapshot"]].copy()
        bad = bad.sort_values("price_at_snapshot", ascending=False)
        print(f"\n  Out-of-range rows (price_at_snapshot > 1.0):")
        print(bad.to_string(index=False))
        print(f"  ACTION NEEDED: clip these to 1.0 before modelling")

    # Rolling feature sanity
    print(f"\n--- Rolling Feature Sanity ---")
    for window in ["7d", "14d"]:
        vol_col = f"price_volatility_{window}"
        chg_col = f"price_change_{window}"
        if vol_col in df.columns:
            v = df[vol_col].dropna()
            c = df[chg_col].dropna()
            print(f"  volatility_{window}: min={v.min():.4f}  max={v.max():.4f}  "
                  f"negative={(v < 0).sum()}  <- should be 0")
            print(f"  change_{window}:     min={c.min():.4f}  max={c.max():.4f}")

    # Train/test split
    print(f"\n--- Train/Test Split ---")
    for split in ["train", "test"]:
        sub     = df[df["split"] == split]
        yes_pct = sub["outcome"].mean()
        n_mkts  = sub["market_id"].nunique()
        print(f"  {split}: {len(sub):>8,} rows | {n_mkts:>5,} markets | "
              f"YES={yes_pct:.1%} NO={1 - yes_pct:.1%}")

    # Leakage check
    train_ids = set(df[df["split"] == "train"]["market_id"].unique())
    test_ids  = set(df[df["split"] == "test"]["market_id"].unique())
    overlap   = train_ids & test_ids
    print(f"\n  Markets in BOTH train and test: {len(overlap)}  <- must be 0")
    if overlap:
        print(f"  WARNING — leakage detected: {list(overlap)[:5]}")

    # Snapshot timestamp range
    print(f"\n--- Snapshot Timestamp Range ---")
    print(f"  Earliest: {df['snapshot_timestamp'].min()}")
    print(f"  Latest:   {df['snapshot_timestamp'].max()}")

    # ── Data quality summary ───────────────────
    section("DATA QUALITY SUMMARY")
    print("  Issues requiring attention:")
    if meta_dupe_count > 0:
        print(f"  1. {meta_dupe_count} duplicate market_ids in meta — deduplicate before modelling")
    else:
        print(f"  1. No duplicate market_ids in meta  ✓")
    if out_of_range > 0:
        print(f"  2. {out_of_range} price_at_snapshot > 1.0 (max={p.max():.4f}) — clip to 1.0 before modelling")
    else:
        print(f"  2. All prices within [0, 1]  ✓")
    print(f"  3. ~{nan_7d_mask.mean():.0%} of snapshots have NaN 7d rolling features "
          f"(early lifecycle; expected — drop or ffill for models that can't handle NaN)")
    cat_nan_pct = df["category"].isna().mean()
    print(f"  4. category is {cat_nan_pct:.1%} missing in dataset — run categorize_markets.py first")

    # ── Plots ──────────────────────────────────

    # Price distribution by outcome
    fig, ax = plt.subplots(figsize=(9, 4))
    prices_clipped = df["price_at_snapshot"].clip(0, 1)
    for outcome_val, label, color in [(0, "NO (0)", "#F44336"), (1, "YES (1)", "#2196F3")]:
        subset = prices_clipped[df["outcome"] == outcome_val]
        ax.hist(subset, bins=50, alpha=0.65, label=label, color=color, density=True)
    ax.set_xlabel("price_at_snapshot (implied P(YES))")
    ax.set_ylabel("Density")
    ax.set_title("Price Distribution at Snapshot by Final Outcome")
    ax.legend()
    plt.tight_layout()
    save_fig("dataset_price_distribution.png")

    # Snapshots per market
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(snaps_per.values, bins=60, color="#9C27B0", edgecolor="white")
    ax.axvline(snaps_per.median(), color="red", linestyle="--",
               label=f"Median = {snaps_per.median():.0f}")
    ax.set_xlabel("Snapshots per Market")
    ax.set_ylabel("Number of Markets")
    ax.set_title("Snapshots per Market Distribution")
    ax.legend()
    plt.tight_layout()
    save_fig("dataset_snapshots_per_market.png")

    # NaN fraction by lifecycle stage
    df_tmp = df[["pct_lifetime_elapsed", ROLLING_7D[0]]].copy()
    df_tmp["pct_bucket"] = pd.cut(
        df_tmp["pct_lifetime_elapsed"], bins=10,
        labels=[f"{i*10}–{(i+1)*10}%" for i in range(10)],
    )
    nan_frac = (
        df_tmp.groupby("pct_bucket", observed=True)[ROLLING_7D[0]]
        .apply(lambda x: x.isna().mean())
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    nan_frac.plot(kind="bar", ax=ax, color="#FF5722", edgecolor="white")
    ax.set_xlabel("% of Market Lifetime Elapsed")
    ax.set_ylabel("Fraction of Snapshots with NaN (7d features)")
    ax.set_title("NaN Rolling Features by Lifecycle Stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_fig("feature_nan_by_lifecycle.png")


# ─────────────────────────────────────────────
# CALIBRATION (RQ1 BASELINE)
# ─────────────────────────────────────────────

def analyze_calibration(df: pd.DataFrame) -> None:
    """Compute market price calibration on train split."""
    section("MARKET PRICE CALIBRATION (RQ1 BASELINE)")

    train = df[df["split"] == "train"].copy()
    train["price_at_snapshot"] = train["price_at_snapshot"].clip(0, 1)
    clean   = train.dropna(subset=["price_at_snapshot", "outcome"])
    dropped = len(train) - len(clean)
    print(f"  Train rows used: {len(clean):,}  (dropped {dropped:,} NaN rows)")

    y_true  = clean["outcome"].values
    y_score = clean["price_at_snapshot"].values
    y_pred  = (y_score > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    ll  = log_loss(y_true, y_score)

    print(f"\n  Baseline (price_at_snapshot as predictor, TRAIN only):")
    print(f"  Accuracy (threshold=0.5): {acc:.4f}")
    print(f"  AUC-ROC:                  {auc:.4f}")
    print(f"  Log-loss:                 {ll:.4f}")
    print(f"\n  Interpretation: AUC-ROC near 1.0 means market price is already highly")
    print(f"  predictive. ML models must beat this baseline to add value (RQ1).")

    # Calibration by price bin
    bins   = np.linspace(0, 1, 11)
    labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(10)]
    clean = clean.copy()
    clean["price_bin"] = pd.cut(
        clean["price_at_snapshot"], bins=bins,
        labels=labels, include_lowest=True,
    )
    cal = (
        clean.groupby("price_bin", observed=True)
        .agg(frac_yes=("outcome", "mean"), count=("outcome", "size"))
        .reset_index()
    )

    print(f"\n  Calibration by price bin (train):")
    print(f"  {'Bin':<12} {'Frac YES':>10} {'Count':>12}")
    for _, row in cal.iterrows():
        print(f"  {str(row['price_bin']):<12} {row['frac_yes']:>10.3f} {row['count']:>12,}")

    # Calibration plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    frac_yes    = cal["frac_yes"].values
    counts      = cal["count"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.plot(bin_centers, frac_yes, color="#2196F3", linewidth=1.5)
    size = np.maximum(counts / counts.max() * 300, 20)
    ax.scatter(bin_centers, frac_yes, s=size, color="#2196F3", alpha=0.85,
               label="Observed frequency", zorder=5)
    ax.set_xlabel("Market Price at Snapshot (Implied P(YES))")
    ax.set_ylabel("Fraction Resolving YES")
    ax.set_title(f"Calibration Curve (Train)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    save_fig("calibration_curve.png")


# ─────────────────────────────────────────────
# CLASS IMBALANCE PLOT
# ─────────────────────────────────────────────

def plot_class_imbalance(meta: pd.DataFrame, df: pd.DataFrame) -> None:
    market_counts = meta["outcome"].map({1: "YES", 0: "NO"}).value_counts()
    snap_counts   = df["outcome"].map({1: "YES", 0: "NO"}).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, counts, title, total in [
        (axes[0], market_counts, "Market Level (Meta)",    len(meta)),
        (axes[1], snap_counts,   "Snapshot Level (Dataset)", len(df)),
    ]:
        colors = ["#2196F3" if k == "YES" else "#F44336" for k in counts.index]
        bars   = ax.bar(counts.index, counts.values, color=colors)
        for bar, v in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{v:,}\n({v / total:.1%})",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylabel("Count")

    fig.suptitle("Class Imbalance: YES vs NO Outcomes", fontsize=13)
    plt.tight_layout()
    save_fig("class_imbalance.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  POLYMARKET — DATA EXPLORATION & VALIDATION")
    print("=" * 60)

    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found. Run fetch_markets.py first.")
        return
    if not DATASET_CSV.exists():
        print(f"ERROR: {DATASET_CSV} not found. Run build_snapshots.py first.")
        return

    meta = pd.read_csv(META_CSV)
    meta["start_date"] = pd.to_datetime(meta["start_date"], format="ISO8601", utc=True)
    meta["end_date"]   = pd.to_datetime(meta["end_date"],   format="ISO8601", utc=True)

    print("\nLoading dataset parquet (4M rows)...")
    df = pd.read_parquet(DATASET_CSV)

    dupe_count = explore_meta(meta)
    explore_dataset(df, meta_dupe_count=dupe_count)
    analyze_calibration(df)
    plot_class_imbalance(meta, df)

    section("DONE")
    print(f"  All plots saved to ./{PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()
