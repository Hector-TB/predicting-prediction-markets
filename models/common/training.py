"""
Shared training protocol for every model (ADR-024).

    1. fit / holdout: train markets sorted by resolution time; the oldest 80% are
       `fit`, the newest 20% are `holdout`. Holdout rows dated before the last fit
       market resolved are dropped — the same rule as the train/test split (ADR-021).
    2. Weights: each market counts once in total (1 / its snapshot count), on top
       of the library's class balancing (ADR-007).
    3. Settings are chosen by AUC on the holdout; the final model is fitted on `fit`.
    4. Isotonic calibration and the decision threshold come from the holdout.
    5. The test set is scored once, at the end.

Import pattern (from any models/<name>/train.py):
    from models.common.training import load_dataset, split_holdout, market_weights, fit_calibrator
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TARGET   = "outcome"

HOLDOUT_FRACTION = 0.2
# Calibrated probabilities never reach exactly 0 or 1: a holdout group that went
# 53-for-53 is very likely, not certain (ADR-024)
PROB_FLOOR, PROB_CEIL = 0.001, 0.999


def load_dataset(trends: bool = False) -> pd.DataFrame:
    """The leakage-filtered dataset (ADR-005/014); the trends file has the same rows plus trend columns."""
    name = "polymarket_ml_dataset_with_trends_clean.parquet" if trends else "polymarket_ml_dataset_clean.parquet"
    log.info("Loading %s ...", name)
    df = pd.read_parquet(DATA_DIR / name)
    df["category"] = df["category"].fillna("other")
    df = df.dropna(subset=[TARGET])
    log.info("  Total rows: %s | train: %s | test: %s",
             f"{len(df):,}", f"{(df['split'] == 'train').sum():,}", f"{(df['split'] == 'test').sum():,}")
    return df


def resolution_times(market_ids) -> pd.Series:
    """
    When each market resolved: Gamma's closedTime, else the scheduled end_date —
    the same rule scripts/recompute_split.py uses for the train/test split.
    """
    ids = pd.Index(pd.unique(np.asarray(market_ids)))
    closed = pd.read_csv(DATA_DIR / "market_closed_times.csv").set_index("market_id")["closed_time"]
    end = pd.read_csv(DATA_DIR / "polymarket_markets_meta.csv", usecols=["market_id", "end_date"]).set_index("market_id")["end_date"]
    resolved = pd.to_datetime(closed.reindex(ids), format="ISO8601", utc=True)
    resolved = resolved.fillna(pd.to_datetime(end.reindex(ids), format="ISO8601", utc=True))
    missing = resolved.isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} markets have neither closedTime nor end_date — "
                         "pull the full dataset version (data/sync.py pull)")
    return resolved


def split_holdout(train: pd.DataFrame, fraction: float = HOLDOUT_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train into (fit, holdout) by resolution time, mirroring the train/test split.

    Holdout rows dated before the last fit market resolved are dropped, so nothing
    the model learned from postdates a holdout snapshot.
    """
    resolved = resolution_times(train["market_id"]).sort_values(kind="stable")
    n_fit = int(len(resolved) * (1 - fraction))
    fit_ids = set(resolved.index[:n_fit])
    cutoff = resolved.iloc[n_fit - 1]

    in_fit = train["market_id"].isin(fit_ids)
    fit = train[in_fit].reset_index(drop=True)
    holdout = train[~in_fit & (train["snapshot_timestamp"] >= cutoff)].reset_index(drop=True)

    dropped = int((~in_fit).sum()) - len(holdout)
    log.info("  Holdout cutoff (last fit resolution): %s", cutoff)
    log.info("  fit:     %s rows | %s markets", f"{len(fit):,}", f"{fit['market_id'].nunique():,}")
    log.info("  holdout: %s rows | %s markets  (%s pre-cutoff rows dropped)",
             f"{len(holdout):,}", f"{holdout['market_id'].nunique():,}", f"{dropped:,}")
    log.info("  YES rate — fit: %.3f | holdout: %.3f", fit[TARGET].mean(), holdout[TARGET].mean())
    return fit, holdout


def market_weights(market_ids) -> np.ndarray:
    """1 / (the market's snapshot count), scaled to mean 1, so every market counts once in total."""
    ids = pd.Series(np.asarray(market_ids))
    w = 1.0 / ids.map(ids.value_counts()).to_numpy(dtype=np.float64)
    return (w / w.mean()).astype(np.float32)


def fit_calibrator(p_holdout: np.ndarray, y_holdout: np.ndarray) -> IsotonicRegression:
    """Isotonic map from raw model scores to probabilities in [PROB_FLOOR, PROB_CEIL], fitted on the holdout."""
    iso = IsotonicRegression(y_min=PROB_FLOOR, y_max=PROB_CEIL, out_of_bounds="clip")
    iso.fit(p_holdout, y_holdout)
    return iso
