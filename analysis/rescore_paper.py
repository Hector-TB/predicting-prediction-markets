"""
Re-score the course paper's comparisons (docs/paper/paper.tex) on the current data.

    python analysis/rescore_paper.py                      # print to the log
    python analysis/rescore_paper.py --out results.md     # also write markdown
    python analysis/rescore_paper.py --n-boot 500         # tighter CIs, slower

Every model is scored on the SAME test snapshots as the market baseline: the
inner join of all available prediction files with the clean parquet. (The
paper's baseline and models were scored on slightly different row sets.)

Sections, mirroring the paper:
  RQ1  — AUC / PR-AUC / log-loss / Brier vs the market price, with
         market-clustered bootstrap CIs on the AUC, log-loss and Brier
         differences (snapshots within a market are correlated, so rows are not
         independent). Twice: every snapshot counted, and each market counted
         once (weight 1 / its snapshot count), so long markets don't dominate.
  RQ3  — AUC by lifecycle third and by market duration. Lifecycle thirds are
         computed two ways: by each market's snapshot order (what the paper's
         figure used — analysis.ipynb cell 21) and by pct_lifetime_elapsed
         (what the paper's text describes).
  Time left — AUC by days before the scheduled close. v3 kept the last 14 days
         before close (ADR-022); this shows whether the models still add
         anything there, where the market price is most informed. Then split by
         market duration, since only longer markets have snapshots ≥ 30 days out.
  5.5  — Trading simulation, two ways:
           per-snapshot     — the paper's method: every snapshot where
                              |p_model − p_market| > τ is a separate trade.
           one per market   — only the first qualifying snapshot of each market,
                              so correlated repeat bets don't inflate the result.
         Both against an always-buy-NO baseline, over trade thresholds τ and a
         cost per token (spread/fees: YES costs p + c, NO costs 1 − p + c).

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
from models.common.evaluation import (  # noqa: E402
    bootstrap_auc_diff, bootstrap_loss_diff, compute_metrics, dataset_version,
)
from models.common.training import market_weights  # noqa: E402

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
TAUS = [0.02, 0.05, 0.10]
COSTS = [0.0, 0.01, 0.02]  # per token: spread + fees

LIFECYCLE = [("Far (0–33%)", 0, 1 / 3), ("Mid (33–67%)", 1 / 3, 2 / 3), ("Near (67–100%)", 2 / 3, 1.01)]
DURATION = [("≤ 90d", 0, 90), ("91–180d", 91, 180), ("181–365d", 181, 365), ("> 365d", 366, 10**6)]
DURATION_WIDE = [("≤ 90d", 0, 90), ("91–365d", 91, 365), ("> 365d", 366, 10**6)]
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

def ci(d: dict, fmt: str = "+.4f") -> str:
    return f"{d['point']:{fmt}} [{d['ci_lo']:{fmt}}, {d['ci_hi']:{fmt}}]"


def rq1(df: pd.DataFrame, labels: list[str], n_boot: int, per_market: bool = False) -> str:
    y = df["outcome"].values
    g = df["market_id"].values
    w = market_weights(g) if per_market else None
    base = df[MARKET].values

    rows, diffs = [], []
    for label in [MARKET] + labels:
        s = compute_metrics(y, df[label].values, sample_weight=w)
        rows.append([label, f"{s['auc']:.4f}", f"{s['pr_auc']:.4f}", f"{s['log_loss']:.4f}", f"{s['brier']:.4f}"])
        if label == MARKET:
            continue
        a = bootstrap_auc_diff(y, base, df[label].values, n_boot=n_boot, groups=g, verbose=False, sample_weight=w)
        l = bootstrap_loss_diff(y, base, df[label].values, groups=g, n_boot=n_boot, sample_weight=w)
        a = {"point": a["point"], "ci_lo": a["ci_lo"], "ci_hi": a["ci_hi"]}
        diffs.append([label, ci(a), ci(l["log_loss"]), ci(l["brier"])])

    if per_market:
        title = "## RQ1 — each market counted once"
        intro = ("Same snapshots, each weighted by 1 / its market's snapshot count, so every market counts once "
                 "(as in training, ADR-024). Shows whether the result holds across typical markets or rests on long ones.")
    else:
        title = "## RQ1 — models vs the market price"
        intro = f"{len(df):,} test snapshots, {df['market_id'].nunique():,} markets. Every snapshot counted."
    return (f"{title}\n\n{intro}\n\n"
            + table(["Model", "AUC-ROC", "PR-AUC", "Log-loss", "Brier"], rows)
            + f"\n\nDifferences vs the market (model − market), 95% CI from {n_boot} market-level bootstrap resamples. "
              "Higher AUC is better; **lower (negative) log-loss and Brier are better**.\n\n"
            + table(["Model", "ΔAUC [95% CI]", "ΔLog-loss [95% CI]", "ΔBrier [95% CI]"], diffs))


def by_bucket(df: pd.DataFrame, labels: list[str], title: str, col: str, buckets) -> str:
    rows = []
    for name, lo, hi in buckets:
        sub = df[(df[col] >= lo) & ((df[col] < hi) if col in HALF_OPEN else (df[col] <= hi))]
        y = sub["outcome"].values
        rows.append([name, f"{len(sub):,}", f"{sub['market_id'].nunique():,}", auc_or_blank(y, sub[MARKET].values)]
                    + [auc_or_blank(y, sub[l].values) for l in labels])
    return f"## {title}\n\nAUC-ROC per bucket.\n\n" + table(["Bucket", "Snapshots", "Markets", MARKET] + labels, rows)


def time_left_by_duration(df: pd.DataFrame, labels: list[str]) -> str:
    parts = ["## Time left × market duration\n\n"
             "Only markets longer than 30 days can have snapshots ≥ 30 days before close, so the time-left table mixes "
             "\"early\" with \"long market\". Here it is split by duration. Cells: market AUC, then each model's AUC "
             "minus the market's. Buckets with few markets are noisy."]
    for dname, dlo, dhi in DURATION_WIDE:
        sub_d = df[(df["duration_days"] >= dlo) & (df["duration_days"] <= dhi)]
        rows = []
        for tname, tlo, thi in TIME_LEFT:
            sub = sub_d[(sub_d["days_before_close"] >= tlo) & (sub_d["days_before_close"] < thi)]
            y = sub["outcome"].values
            if len(sub) == 0 or len(np.unique(y)) < 2:
                rows.append([tname, f"{len(sub):,}", f"{sub['market_id'].nunique():,}", "—"] + ["—"] * len(labels))
                continue
            m = compute_metrics(y, sub[MARKET].values)["auc"]
            rows.append([tname, f"{len(sub):,}", f"{sub['market_id'].nunique():,}", f"{m:.4f}"]
                        + [f"{compute_metrics(y, sub[l].values)['auc'] - m:+.4f}" for l in labels])
        parts.append(f"### Duration {dname}\n\n"
                     + table(["Time left", "Snapshots", "Markets", "Market AUC"] + [f"Δ {l}" for l in labels], rows))
    return "\n\n".join(parts)


def simulate(df: pd.DataFrame, p_model: np.ndarray | None, one_per_market: bool,
             n_boot: int, tau: float = TAU, cost: float = 0.0, seed: int = 42) -> dict:
    """
    Buy YES when the model is > tau above the market, NO when > tau below.
    p_model=None means always buy NO. One token per trade; each token costs its
    price plus `cost` (YES: p + c, NO: 1 − p + c) and pays 1 if right.
    """
    p = df["price_at_snapshot"].values
    y = df["outcome"].values
    if p_model is None:
        trade, yes = np.ones(len(df), bool), np.zeros(len(df), bool)
    else:
        edge = p_model - p
        trade, yes = np.abs(edge) > tau, edge > 0

    t = pd.DataFrame({
        "market_id": df["market_id"].values,
        "ts":        df["snapshot_timestamp"].values,
        "profit":    np.where(yes, y - p, p - y) - cost,
        "capital":   np.where(yes, p, 1 - p) + cost,
    })[trade]
    if one_per_market:
        t = t.sort_values("ts").groupby("market_id", sort=False).head(1)
    if t.empty:
        return {"trades": 0, "markets": 0, "profit": 0.0, "roi": np.nan, "win": np.nan, "lo": np.nan, "hi": np.nan}

    # ROI CI: resample markets (ratio of summed profit to summed capital)
    per_mkt = t.groupby("market_id")[["profit", "capital"]].sum().values
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(per_mkt), len(per_mkt)), minlength=len(per_mkt))
                       for _ in range(n_boot)])
    sums = counts @ per_mkt
    lo, hi = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5])
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
             f"Paper's rule: trade when |p_model − p_market| > {TAU}; one token per trade; no fees or slippage. "
             "ROI = total profit / total capital; 95% CI from market-level bootstrap."]
    for one, name in [(False, "Per-snapshot (paper's method)"), (True, "One trade per market (first qualifying snapshot)")]:
        rows = []
        for label, pm in [("Always buy NO", None)] + [(l, df[l].values) for l in labels]:
            r = simulate(df, pm, one, n_boot)
            rows.append([label, f"{r['trades']:,}", f"{r['markets']:,}", f"${r['profit']:,.0f}",
                         f"{r['roi']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]", f"{r['win']:.1%}"])
        parts.append(f"### {name}\n\n" + table(["Strategy", "Trades", "Markets", "Profit", "ROI [95% CI]", "Win rate"], rows))

    parts.append("## Trading with costs and other thresholds\n\n"
                 "Each token costs its price plus c (spread and fees): YES costs p + c, NO costs 1 − p + c. "
                 "τ = minimum gap between model and market to trade. Cells: ROI [95% CI]; trade counts don't depend on c.")
    header = ["Strategy", "τ", "Trades"] + [f"c = {c:.2f}" for c in COSTS]
    for one, name in [(False, "Per snapshot"), (True, "One trade per market")]:
        rows = []
        strategies = [("Always buy NO", None, None)] + [(l, df[l].values, tau) for l in labels for tau in TAUS]
        for label, pm, tau in strategies:
            res = [simulate(df, pm, one, n_boot, tau=tau or TAU, cost=c) for c in COSTS]
            rows.append([label, "—" if tau is None else f"{tau:.2f}", f"{res[0]['trades']:,}"]
                        + [f"{r['roi']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]" for r in res])
        parts.append(f"### {name}\n\n" + table(header, rows))
    return "\n\n".join(parts)


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
        rq1(df, labels, args.n_boot, per_market=True),
        by_bucket(df, labels, "RQ3 — lifecycle stage (thirds of each market's snapshots, as in the paper's figure)", "rank_pct", LIFECYCLE),
        by_bucket(df, labels, "RQ3 — lifecycle stage (by pct_lifetime_elapsed)", "pct_lifetime_elapsed", LIFECYCLE),
        by_bucket(df, labels, "RQ3 — market duration", "duration_days", DURATION),
        by_bucket(df, labels, "Time left before the scheduled close (v3 adds the last 14 days, ADR-022)",
                  "days_before_close", TIME_LEFT),
        time_left_by_duration(df, labels),
        trading(df, labels, args.n_boot),
    ]) + "\n"

    log.info("\n%s", report)
    if args.out:
        args.out.write_text(report)
        log.info("Wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
