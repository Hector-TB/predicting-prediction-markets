"""
Snapshot features — one implementation for training and serving
================================================================
Training builds features in two steps: build_snapshots.py computes the raw
features for each snapshot (`raw_features`), then fix_dataset.py clips the
price and fills empty rolling windows (`finish_features`). The API must apply
exactly the same steps to a live market, so both live here:

    price_df = fetch_price_history(clob_token_id)
    row      = features_at(price_df, now, start_date, end_date, duration_days)

`scripts/check_feature_parity.py` checks that `features_at` reproduces the
published training snapshots.
"""

import time
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

CLOB_URL        = "https://clob.polymarket.com"
MIN_PRICE_OBS   = 20        # minimum price observations, per market and per snapshot
ROLLING_WINDOWS = [7, 14]   # rolling feature windows in days
MAX_RETRIES     = 5
RETRY_BACKOFF   = 5         # seconds before first retry (doubles each attempt)

ROLLING_STATS      = ["mean", "volatility", "min", "max", "change", "range", "trend"]
ROLLING_FILL_PRICE = ["price_mean", "price_min", "price_max"]
ROLLING_FILL_ZERO  = ["price_volatility", "price_change", "price_range", "price_trend"]


class FetchError(Exception):
    """The price history could not be fetched (network / server error), as
    opposed to the market genuinely having no history."""


def fetch_price_history(clob_token_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch full price history for a YES token from CLOB API.
    Returns DataFrame(timestamp, price) sorted ascending, or None if the market
    has no (or too little) history. Raises FetchError if the request keeps
    failing, so a network blip is never mistaken for "no history".
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                f"{CLOB_URL}/prices-history",
                params={"market": clob_token_id, "interval": "max", "fidelity": 720},
                timeout=20,
            )
            r.raise_for_status()
            history = r.json().get("history", [])
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise FetchError(str(e)) from e
            time.sleep(RETRY_BACKOFF * 2 ** (attempt - 1))

    if not history:
        return None

    df = pd.DataFrame(history)
    df = df.rename(columns={"t": "timestamp", "p": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["price"]     = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values("timestamp").reset_index(drop=True)

    return df if len(df) >= MIN_PRICE_OBS else None


def raw_features(hist: pd.DataFrame, snap_ts: pd.Timestamp, start: pd.Timestamp,
                 end: pd.Timestamp, duration_days: int) -> dict:
    """Features at `snap_ts` from `hist`, the price history up to and including
    `snap_ts` (at least MIN_PRICE_OBS rows). Rolling windows with fewer than two
    prices are NaN; `finish_features` fills them."""
    price_now         = float(hist["price"].values[-1])
    days_before_close = (end - snap_ts).total_seconds() / 86400
    pct_elapsed       = float(np.clip(
        (snap_ts - start).total_seconds() / (end - start).total_seconds(), 0, 1
    ))

    feat = {
        "days_before_close":         round(days_before_close, 2),
        "pct_lifetime_elapsed":      round(pct_elapsed, 4),
        "duration_days":             duration_days,
        "price_at_snapshot":         round(price_now, 4),
        "price_deviation_from_half": round(abs(price_now - 0.5), 4),
    }

    for window_days in ROLLING_WINDOWS:
        label        = f"{window_days}d"
        window_start = snap_ts - timedelta(days=window_days)
        w_prices     = hist[hist["timestamp"] >= window_start]["price"].values

        if len(w_prices) >= 2:
            x     = np.arange(len(w_prices), dtype=float)
            slope = float(np.polyfit(x, w_prices, 1)[0]) if x.std() > 0 else 0.0
            feat[f"price_mean_{label}"]       = round(float(np.mean(w_prices)), 4)
            feat[f"price_volatility_{label}"] = round(float(np.std(w_prices)), 4)
            feat[f"price_min_{label}"]        = round(float(np.min(w_prices)), 4)
            feat[f"price_max_{label}"]        = round(float(np.max(w_prices)), 4)
            feat[f"price_change_{label}"]     = round(float(w_prices[-1] - w_prices[0]), 4)
            feat[f"price_range_{label}"]      = round(float(np.max(w_prices) - np.min(w_prices)), 4)
            feat[f"price_trend_{label}"]      = round(slope, 6)
        else:
            for s in ROLLING_STATS:
                feat[f"price_{s}_{label}"] = np.nan

    return feat


def finish_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip the price to [0, 1] and fill empty rolling windows (in place):
    price_mean/min/max → price_at_snapshot (no trading = flat price),
    volatility/change/range/trend → 0.0."""
    df["price_at_snapshot"] = df["price_at_snapshot"].clip(0.0, 1.0)
    df["price_deviation_from_half"] = (df["price_at_snapshot"] - 0.5).abs().round(4)

    for w in ROLLING_WINDOWS:
        nan_mask = df[f"price_mean_{w}d"].isna()
        for feat in ROLLING_FILL_PRICE:
            df.loc[nan_mask, f"{feat}_{w}d"] = df.loc[nan_mask, "price_at_snapshot"].round(4)
        for feat in ROLLING_FILL_ZERO:
            df.loc[nan_mask, f"{feat}_{w}d"] = 0.0
    return df


def features_at(price_df: pd.DataFrame, ts: pd.Timestamp, start: pd.Timestamp,
                end: pd.Timestamp, duration_days: int) -> Optional[dict]:
    """The finished feature row at `ts`, as training sees it, from prices up to
    and including `ts`; None if there are fewer than MIN_PRICE_OBS of them."""
    hist = price_df[price_df["timestamp"] <= ts]
    if len(hist) < MIN_PRICE_OBS:
        return None
    df = finish_features(pd.DataFrame([raw_features(hist, ts, start, end, duration_days)]))
    return df.iloc[0].to_dict()
