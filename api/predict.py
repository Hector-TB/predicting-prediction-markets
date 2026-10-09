"""
Predict one open market with the production release (ADR-017, ADR-025)
======================================================================
Features come from `features.features_at`, the code that built the training
snapshots. Every open binary market gets a prediction; inputs training never
saw are filled for the models only and named in a warning.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

import features
from api import gamma
from api.releases import ServingRelease
from categorize_markets import classify_batch
from fetch_markets import MIN_DURATION_DAYS, VOLUME_NUM_MIN, parse_dt, parse_json_field

log = logging.getLogger(__name__)

BURN_IN_DAYS    = 14     # training's first snapshot: createdAt + 14 days (ADR-002)
SETTLED         = 0.95   # training drops rows priced >= 0.95 or <= 0.05 (ADR-014)
NEAR_CLOSE_DAYS = 30     # no edge over the market measured closer to close (v3 re-score)
LONG_DAYS       = 365    # no edge measured in markets longer than a year

WARNINGS = {
    "no_end_date":     "No usable end date: the time features are set to typical training values. "
                       "Rankings held up in testing, but probabilities were less accurate than the market price.",
    "past_end_date":   "Past its scheduled end date and not yet resolved: treated as 0 days left.",
    "few_prices":      "Fewer than {min_obs} price points; the models were trained on markets with at least {min_obs}.",
    "young_market":    "Less than {burn_in} days old; training snapshots start {burn_in} days after creation.",
    "settled_price":   "Price is at or beyond {hi} / {lo}; training left out such near-settled snapshots.",
    "near_close":      "Less than {near} days to close; the models showed no edge over the market this close.",
    "long_market":     "Scheduled to run more than a year; the models showed no edge in such markets.",
    "outside_filters": "Shorter than {min_dur} days or under ${min_vol:,} volume: outside the markets used for training.",
    "no_category":     "Category could not be determined; the models used 'other'.",
}
FORMAT = {"min_obs": features.MIN_PRICE_OBS, "burn_in": BURN_IN_DAYS, "hi": SETTLED,
          "lo": round(1 - SETTLED, 2), "near": NEAR_CLOSE_DAYS, "min_dur": MIN_DURATION_DAYS,
          "min_vol": VOLUME_NUM_MIN}


class PredictError(Exception):
    """A market that can't be predicted: not_found, not_binary, closed or no_price."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Market:
    market_id: str
    question: str
    slug: Optional[str]
    event_slug: Optional[str]
    token: Optional[str]
    start: pd.Timestamp
    end: Optional[pd.Timestamp]       # None if missing or not after start
    end_raw: Optional[str]            # Gamma's endDate as seen now, stored with the lookup
    volume: float
    price: Optional[float]

    @property
    def duration_days(self) -> Optional[int]:
        return (self.end - self.start).days if self.end is not None else None

    @property
    def url(self) -> Optional[str]:
        if self.event_slug and self.slug:
            return f"https://polymarket.com/event/{self.event_slug}/{self.slug}"
        return f"https://polymarket.com/market/{self.slug}" if self.slug else None


def parse_market(m: dict) -> Market:
    if not gamma.is_binary(m):
        raise PredictError("not_binary", "Only YES/NO markets can be predicted.")
    if m.get("closed"):
        raise PredictError("closed", "This market has closed.")
    start = parse_dt(m.get("createdAt"))      # training's market start (fetch_markets.py)
    end   = parse_dt(m.get("endDate"))
    if start is None:
        raise PredictError("no_start_date", "The market has no creation date.")
    events = m.get("events") or []
    tokens = parse_json_field(m.get("clobTokenIds"), [])
    return Market(
        market_id=m["conditionId"], question=m.get("question", ""), slug=m.get("slug"),
        event_slug=events[0].get("slug") if events else None,
        token=tokens[0] if tokens else None, start=start,
        end=end if end is not None and end > start else None, end_raw=m.get("endDate"),
        volume=float(m.get("volumeNum") or 0), price=gamma.yes_price(m),
    )


# ─────────────────────────────────────────────
# CATEGORY (one Claude Haiku call per market, cached)
# ─────────────────────────────────────────────

_categories: dict[str, str] = {}
_client: Optional[anthropic.Anthropic] = None


def category(market: Market) -> Optional[str]:
    """The LLM category, as categorize_markets.py assigns it in training; None on failure."""
    global _client
    if market.market_id in _categories:
        return _categories[market.market_id]
    try:
        _client = _client or anthropic.Anthropic()
        cats = classify_batch(_client, [{"id": market.market_id, "question": market.question}])
    except Exception as e:
        log.warning("category lookup failed for %s: %s", market.market_id, e)
        return None
    cat = cats[0] if cats else None
    if cat:
        _categories[market.market_id] = cat
    return cat


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def warning(code: str) -> dict:
    return {"code": code, "message": WARNINGS[code].format(**FORMAT)}


def build_features(market: Market, price_df: Optional[pd.DataFrame], now: pd.Timestamp,
                   release: ServingRelease) -> tuple[dict, list[str]]:
    """The model inputs as of `now` and the warning codes (ADR-025)."""
    codes = []
    n_prices = 0 if price_df is None else int((price_df["timestamp"] <= now).sum())
    if n_prices == 0:
        if market.price is None:
            raise PredictError("no_price", "The market has no price yet.")
        # No price history: the current price, with flat rolling windows (finish_features)
        price_df = pd.DataFrame({"timestamp": [now], "price": [market.price]})
    if n_prices < features.MIN_PRICE_OBS:
        codes.append("few_prices")

    feats = features.features_at(price_df, now, market.start, market.end,
                                 market.duration_days if market.end is not None else np.nan, min_obs=1)
    if market.end is None:
        feats.update(release.time_fill)
        codes.append("no_end_date")
    elif market.end <= now:
        feats["days_before_close"] = 0.0      # pct_lifetime_elapsed is already clipped to 1
        codes.append("past_end_date")
    elif feats["days_before_close"] < NEAR_CLOSE_DAYS:
        codes.append("near_close")

    if (now - market.start).total_seconds() < BURN_IN_DAYS * 86400:
        codes.append("young_market")
    if feats["price_at_snapshot"] >= SETTLED or feats["price_at_snapshot"] <= 1 - SETTLED:
        codes.append("settled_price")
    duration = market.duration_days
    if duration is not None and duration > LONG_DAYS:
        codes.append("long_market")
    if (duration is not None and duration < MIN_DURATION_DAYS) or market.volume < VOLUME_NUM_MIN:
        codes.append("outside_filters")
    return feats, codes


def predict(market_id: str, release: ServingRelease, now: Optional[pd.Timestamp] = None) -> dict:
    raw = gamma.market_by_condition_id(market_id)
    if raw is None:
        raise PredictError("not_found", f"No Polymarket market with condition ID {market_id}.")
    market = parse_market(raw)
    now = now or pd.Timestamp(datetime.now(timezone.utc))

    price_df = features.fetch_price_history(market.token, min_obs=1) if market.token else None
    feats, codes = build_features(market, price_df, now, release)

    cat = category(market)
    if cat is None:
        codes.append("no_category")
    feats["category"] = cat or "other"

    row = pd.DataFrame([feats])
    preds = []
    for name, (module, artifact) in release.models.items():
        p = float(module.predict(artifact, row)[0])
        preds.append({"model": name, "probability": round(p, 4),
                      "gap": round(p - market.price, 4) if market.price is not None else None})
    preds.sort(key=lambda x: x["model"] != release.headline)

    history = [] if price_df is None else [
        {"t": t.isoformat(), "p": float(p)} for t, p in zip(price_df["timestamp"], price_df["price"])]
    return {
        "market": {
            "market_id": market.market_id, "question": market.question, "category": cat,
            "url": market.url, "start_date": market.start.isoformat(), "end_date": market.end_raw,
            "price": market.price, "volume_to_date": market.volume,
        },
        "as_of": now.isoformat(),
        "release": release.name,
        "dataset_version": release.dataset_version,
        "headline_model": release.headline,
        "predictions": preds,
        "warnings": [warning(c) for c in codes],
        "features": {k: v if isinstance(v, str) else (None if np.isnan(v) else float(v))
                     for k, v in feats.items()},
        "price_history": history,
    }
