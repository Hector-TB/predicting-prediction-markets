"""
Re-score the course paper's comparisons (docs/paper/paper.tex) on the current data.

    python analysis/rescore_paper.py                      # print to the log
    python analysis/rescore_paper.py --out results.md     # also write markdown
    python analysis/rescore_paper.py --n-boot 500         # tighter CIs, slower

Every model is scored on the SAME test snapshots as the market baseline: the
inner join of all available prediction files with the clean parquet. (The
paper's baseline and models were scored on slightly different row sets.)

Sections, mirroring the paper:
  RQ1  — AUC / PR-AUC / log-loss / Brier vs the market price, with a
         market-clustered bootstrap CI on ΔAUC (snapshots within a market are
         correlated, so rows are not independent).
  RQ3  — AUC by lifecycle third and by market duration. Lifecycle thirds are
         computed two ways: by each market's snapshot order (what the paper's
         figure used — analysis.ipynb cell 21) and by pct_lifetime_elapsed
         (what the paper's text describes).
  Time left — AUC by days before the scheduled close. v3 kept the last 14 days
         before close (ADR-022); this shows whether the models still add
         anything there, where the market price is most informed.
  5.5  — Trading simulation, two ways:
           per-snapshot     — the paper's method: every snapshot where
                              |p_model − p_market| > τ is a separate trade.
           one per market   — only the first qualifying snapshot of each market,
                              so correlated repeat bets don't inflate the result.
         Both against an always-buy-NO baseline. No fees or slippage.

Prediction files are matched on (market_id, snapshot_timestamp). A model whose
file lacks the timestamp column (LR/GB before 2026-09-27), or was trained on a
different dataset version than the local data (ADR-016), is skipped.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.common.evaluation import bootstrap_auc_diff, compute_metrics, dataset_version  # noqa: E402

log = logging.getLogger(__name__)

DATA_PATH = ROOT / "data" / "polymarket_ml_dataset_clean.parquet"
MODELS_DIR = ROOT / "models"

LR_DIR = MODELS_DIR / "logistic_regression/predictions"
GB_DIR = MODELS_DIR / "gradient_boosting/predictions"

# label → sources tried in order: (prediction CSV, candidate probability columns).
# The v1 LR/GB files put base + trends in one predictions.csv; the current
# train.py writes predictions.csv and (with --trends) predictions_trends.csv.
MODELS = [
    ("LR",          [(LR_DIR / "predictions.csv", ["pred_prob", "pred_prob_base"])]),
    ("LR + Trends", [(LR_DIR / "predictions_trends.csv", ["pred_prob"]), (LR_DIR / "predictions.csv", ["pred_prob_trends"])]),
    ("GB",          [(GB_DIR / "predictions.csv", ["pred_prob", "pred_prob_base"])]),
    ("GB + Trends", [(GB_DIR / "predictions_trends.csv", ["pred_prob"]), (GB_DIR / "predictions.csv", ["pred_prob_trends"])]),
    ("RF",          [(MODELS_DIR / "random_forest/predictions/test_predictions.csv", ["proba_full_calibrated"])]),
    ("RF + Trends", [(MODELS_DIR / "random_forest_trends/predictions/trends_test_predictions.csv", ["proba_trends_calibrated"])]),
]

KEY = ["market_id", "snapshot_timestamp"]
MARKET = "Market price"
TAU = 0.02  # paper's a-priori trade threshold

LIFECYCLE = [("Far (0–33%)", 0, 1 / 3), ("Mid (33–67%)", 1 / 3, 2 / 3), ("Near (67–100%)", 2 / 3, 1.01)]
DURATION = [("≤ 90d", 0, 90), ("91–180d", 91, 180), ("181–365d", 181, 365), ("> 365d", 366, 10**6)]
TIME_LEFT = [("< 1 day", 0, 1), ("1–7 days", 1, 7), ("7–14 days", 7, 14), ("14–30 days", 14, 30), ("≥ 30 days", 30, 10**6)]
# Buckets that are [lo, hi); the others (whole days) are [lo, hi]
HALF_OPEN = ("pct_lifetime_elapsed", "rank_pct", "days_before_close")


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_test_frame() -> tuple[pd.DataFrame, list[str]]:
    """Test snapshots joined to every model's predictions. Returns (df, model labels)."""
    base = pd.read_parquet(DATA_PATH, columns=KEY + [
        "outcome", "split", "price_at_snapshot", "pct_lifetime_elapsed", "duration_days", "days_before_close", "category",
    ])
    base = base[base["split"] == "test"].drop(columns="split")
    base["snapshot_timestamp"] = pd.to_datetime(base["snapshot_timestamp"], format="mixed", utc=True)
    base[MARKET] = base["price_at_snapshot"]
    log.info("Test snapshots in %s: %s (%s markets)", DATA_PATH.name, f"{len(base):,}", f"{base['market_id'].nunique():,}")

    version = dataset_version()
    labels = []
    cache: dict[Path, pd.DataFrame] = {}
    for label, sources in MODELS:
        path = col = None
        why = "no prediction file found"
        for src, candidates in sources:
            if not src.exists():
                continue
            if src not in cache:
                cache[src] = pd.read_csv(src)
            found = next((c for c in candidates if c in cache[src].columns), None)
            if found is None:
                why = f"no probability column in {src.name}"
            elif "snapshot_timestamp" not in cache[src].columns:
                why = f"{src.name} has no snapshot_timestamp column (retrain)"
            elif set(cache[src].get("dataset_version", pd.Series(["unknown"])).astype(str).unique()) != {version}:
                why = f"{src.name} was not trained on dataset {version} (retrain)"
            else:
                path, col = src, found
                break
        if path is None:
            log.info("  skip %-12s — %s", label, why)
            continue
        preds = cache[path]
        p = preds[KEY + [col]].rename(columns={col: label})
        p["snapshot_timestamp"] = pd.to_datetime(p["snapshot_timestamp"], format="mixed", utc=True)
        before = len(base)
        base = base.merge(p, on=KEY, how="inner")
        log.info("  %-12s ← %s[%s]  (%s rows matched of %s)", label, path.name, col, f"{len(base):,}", f"{before:,}")
        labels.append(label)

    # Position of each snapshot within its own market's test snapshots (0 = first)
    base = base.reset_index(drop=True)
    rank = base.groupby("market_id")["snapshot_timestamp"].rank(method="first")
    base["rank_pct"] = (rank - 1) / base.groupby("market_id")["snapshot_timestamp"].transform("count")

    log.info("Common evaluation set: %s snapshots, %s markets", f"{len(base):,}", f"{base['market_id'].nunique():,}")
    return base, labels


# ─────────────────────────────────────────────
# REPORT HELPERS
# ─────────────────────────────────────────────

def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def auc_or_blank(y, p) -> str:
    return f"{compute_metrics(y, p)['auc']:.4f}" if len(np.unique(y)) == 2 else "—"


# ─────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────

def rq1(df: pd.DataFrame, labels: list[str], n_boot: int) -> str:
    y = df["outcome"].values
    m = compute_metrics(y, df[MARKET].values)
    rows = [[MARKET, f"{m['auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['log_loss']:.4f}", f"{m['brier']:.4f}", "—"]]
    for label in labels:
        s = compute_metrics(y, df[label].values)
        b = bootstrap_auc_diff(y, df[MARKET].values, df[label].values, n_boot=n_boot,
                               groups=df["market_id"].values, verbose=False)
        rows.append([label, f"{s['auc']:.4f}", f"{s['pr_auc']:.4f}", f"{s['log_loss']:.4f}", f"{s['brier']:.4f}",
                     f"{b['point']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"])
    return ("## RQ1 — models vs the market price\n\n"
            f"{len(df):,} test snapshots, {df['market_id'].nunique():,} markets. "
            f"ΔAUC = model − market, 95% CI from {n_boot} market-level bootstrap resamples.\n\n"
            + table(["Model", "AUC-ROC", "PR-AUC", "Log-loss", "Brier", "ΔAUC vs market [95% CI]"], rows))


def by_bucket(df: pd.DataFrame, labels: list[str], title: str, col: str, buckets) -> str:
    rows = []
    for name, lo, hi in buckets:
        sub = df[(df[col] >= lo) & ((df[col] < hi) if col in HALF_OPEN else (df[col] <= hi))]
        y = sub["outcome"].values
        rows.append([name, f"{len(sub):,}", f"{sub['market_id'].nunique():,}", auc_or_blank(y, sub[MARKET].values)]
                    + [auc_or_blank(y, sub[l].values) for l in labels])
    return f"## {title}\n\nAUC-ROC per bucket.\n\n" + table(["Bucket", "Snapshots", "Markets", MARKET] + labels, rows)


def simulate(df: pd.DataFrame, p_model: np.ndarray | None, one_per_market: bool,
             n_boot: int, seed: int = 42) -> dict:
    """
    Buy YES when the model is > TAU above the market, NO when > TAU below.
    p_model=None means always buy NO. One token per trade, no fees.
    """
    p = df["price_at_snapshot"].values
    y = df["outcome"].values
    if p_model is None:
        trade, yes = np.ones(len(df), bool), np.zeros(len(df), bool)
    else:
        edge = p_model - p
        trade, yes = np.abs(edge) > TAU, edge > 0

    t = pd.DataFrame({
        "market_id": df["market_id"].values,
        "ts":        df["snapshot_timestamp"].values,
        "profit":    np.where(yes, y - p, p - y),
        "capital":   np.where(yes, p, 1 - p),
    })[trade]
    if one_per_market:
        t = t.sort_values("ts").groupby("market_id", sort=False).head(1)
    if t.empty:
        return {"trades": 0, "markets": 0, "profit": 0.0, "roi": np.nan, "win": np.nan, "lo": np.nan, "hi": np.nan}

    # ROI CI: resample markets (ratio of summed profit to summed capital)
    per_mkt = t.groupby("market_id")[["profit", "capital"]].sum().values
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = per_mkt[rng.integers(0, len(per_mkt), len(per_mkt))].sum(axis=0)
        boots.append(s[0] / s[1])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {
        "trades":  len(t),
        "markets": t["market_id"].nunique(),
        "profit":  t["profit"].sum(),
        "roi":     t["profit"].sum() / t["capital"].sum(),
        "win":     (t["profit"] > 0).mean(),
        "lo": lo, "hi": hi,
    }


def trading(df: pd.DataFrame, labels: list[str], n_boot: int) -> str:
    parts = ["## Trading simulation (paper §5.5)\n\n"
             f"Trade when |p_model − p_market| > {TAU}; one token per trade; no fees or slippage. "
             "ROI = total profit / total capital; 95% CI from market-level bootstrap."]
    for one, name in [(False, "Per-snapshot (paper's method)"), (True, "One trade per market (first qualifying snapshot)")]:
        rows = []
        for label, pm in [("Always buy NO", None)] + [(l, df[l].values) for l in labels]:
            r = simulate(df, pm, one, n_boot)
            rows.append([label, f"{r['trades']:,}", f"{r['markets']:,}", f"${r['profit']:,.0f}",
                         f"{r['roi']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]", f"{r['win']:.1%}"])
        parts.append(f"### {name}\n\n" + table(["Strategy", "Trades", "Markets", "Profit", "ROI [95% CI]", "Win rate"], rows))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="also write the report as markdown")
    parser.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples (default 200)")
    args = parser.parse_args()

    df, labels = load_test_frame()
    if not labels:
        log.error("No usable prediction files — train the models first.")
        sys.exit(1)

    report = "\n\n".join([
        f"# Paper re-score — {DATA_PATH.name}",
        rq1(df, labels, args.n_boot),
        by_bucket(df, labels, "RQ3 — lifecycle stage (thirds of each market's snapshots, as in the paper's figure)", "rank_pct", LIFECYCLE),
        by_bucket(df, labels, "RQ3 — lifecycle stage (by pct_lifetime_elapsed)", "pct_lifetime_elapsed", LIFECYCLE),
        by_bucket(df, labels, "RQ3 — market duration", "duration_days", DURATION),
        by_bucket(df, labels, "Time left before the scheduled close (v3 adds the last 14 days, ADR-022)",
                  "days_before_close", TIME_LEFT),
        trading(df, labels, args.n_boot),
    ]) + "\n"

    log.info("\n%s", report)
    if args.out:
        args.out.write_text(report)
        log.info("Wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
