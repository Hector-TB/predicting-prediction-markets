"""
Live track record (ADR-017, ADR-026)
===================================
Scores the first prediction of each market, per release and per source
('sample' = the daily random sample, the headline; 'search' = user lookups),
against Gamma's price at that lookup. Metrics only from MIN_RESOLVED resolved
markets with both outcomes; below that, counts only. CIs resample markets (ADR-015).
"""

import numpy as np
import pandas as pd
from release import DB_NAMES  # scripts/release.py: model → model_runs.model_name

from api.predict import BURN_IN_DAYS
from models.common.evaluation import bootstrap_auc_diff, bootstrap_loss_diff, compute_metrics

MIN_RESOLVED = 100
N_BOOT       = 500
BUCKETS      = [("<=90d", 0, 90), ("90-365d", 90, 365), (">365d", 365, np.inf)]   # scheduled duration

# First lookup per (source, market), one row per model prediction
QUERY = """
WITH first AS (
    SELECT DISTINCT ON (source, market_id) id, source, market_id, release, market_price
    FROM snapshots
    WHERE release IS NOT NULL
    ORDER BY source, market_id, snapshot_timestamp, id
)
SELECT f.source, f.market_id, f.release, f.market_price, m.resolution_status, m.outcome,
       m.duration_days, r.model_name, p.probability
FROM first f
JOIN markets m     ON m.market_id = f.market_id
JOIN predictions p ON p.snapshot_id = f.id
JOIN model_runs r  ON r.id = p.model_run_id
"""


def load(cur) -> tuple[pd.DataFrame, dict]:
    """(one row per first lookup with a column per model, {release: {model: per-market backtest metrics}})."""
    cur.execute(QUERY)
    rows = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    model_of = {v: k for k, v in DB_NAMES.items()}
    rows["model"] = rows["model_name"].map(model_of)
    keys = ["source", "market_id", "release", "market_price", "resolution_status", "outcome", "duration_days"]
    wide = (rows.dropna(subset=["model"])
                .pivot_table(index=keys, columns="model", values="probability", dropna=False)
                .reset_index()) if len(rows) else pd.DataFrame(columns=keys)
    wide.columns.name = None

    cur.execute("SELECT release, model_name, metrics FROM model_runs WHERE release IS NOT NULL")
    backtest: dict = {}
    for release, name, metrics in cur.fetchall():
        if name in model_of and metrics and "per_market" in metrics:
            backtest.setdefault(release, {})[model_of[name]] = metrics["per_market"]
    return wide, backtest


def _round(d: dict) -> dict:
    return {k: (_round(v) if isinstance(v, dict) else round(float(v), 5) if k != "n" else int(v))
            for k, v in d.items() if k != "p_value"}


def metrics(df: pd.DataFrame, models: list[str], weight: np.ndarray | None = None) -> dict | None:
    """Market and model metrics with model − market differences, or None if too few markets."""
    y = df["outcome"].to_numpy(dtype=int)
    if len(df) < MIN_RESOLVED or len(np.unique(y)) < 2:
        return None
    market = df["market_price"].to_numpy(dtype=float)
    groups = df["market_id"].to_numpy()
    out = {"n": len(df), "market": _round(compute_metrics(y, market, sample_weight=weight)), "models": {}}
    for m in models:
        p = df[m].to_numpy(dtype=float)
        vs = {"auc": bootstrap_auc_diff(y, market, p, n_boot=N_BOOT, groups=groups, verbose=False,
                                        sample_weight=weight),
              **bootstrap_loss_diff(y, market, p, groups, n_boot=N_BOOT, sample_weight=weight)}
        out["models"][m] = {**_round(compute_metrics(y, p, sample_weight=weight)), "vs_market": _round(vs)}
    return out


def eligible_days(df: pd.DataFrame) -> np.ndarray:
    """Days a market can be drawn by the sample: scheduled duration − burn-in (at least 1)."""
    return np.maximum(df["duration_days"].to_numpy(dtype=float) - BURN_IN_DAYS, 1.0)


def score_group(df: pd.DataFrame, models: list[str], weighted: bool) -> dict:
    status = df["resolution_status"].value_counts()
    done = df[(df["resolution_status"] == "resolved") & df["outcome"].notna()
              & df["market_price"].notna() & df[models].notna().all(axis=1)]
    out = {"markets": len(df), "open": int(status.get("open", 0)),
           "resolved": int(status.get("resolved", 0)), "ambiguous": int(status.get("ambiguous", 0)),
           "metrics": metrics(done, models)}
    if weighted:   # undo the sample's over-representation of long markets (ADR-026)
        timed = done[done["duration_days"].notna()]
        out["metrics_length_weighted"] = metrics(timed, models, 1.0 / eligible_days(timed))
    out["by_duration"] = {}
    for name, lo, hi in BUCKETS:
        b = done[(done["duration_days"] > lo) & (done["duration_days"] <= hi)]
        out["by_duration"][name] = {"resolved": len(b), "metrics": metrics(b, models)}
    return out


def score(wide: pd.DataFrame, backtest: dict) -> dict:
    releases = sorted(set(wide["release"]) | set(backtest), key=lambda r: int(r.lstrip("r")))
    out = []
    for rel in releases:
        df = wide[wide["release"] == rel]
        models = [m for m in DB_NAMES if m in df.columns and df[m].notna().any()] or list(backtest.get(rel, {}))
        df = df.reindex(columns=[*df.columns, *[m for m in models if m not in df.columns]])
        live = {src: score_group(df[df["source"] == src], models, weighted=src == "sample")
                for src in ("sample", "search")}
        out.append({"release": rel, "backtest_per_market": backtest.get(rel, {}), "live": live})
    return {"min_resolved": MIN_RESOLVED, "releases": out}
