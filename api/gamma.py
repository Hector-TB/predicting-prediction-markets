"""Gamma API lookups: turn a URL, slug, condition ID or free text into markets."""

import logging
import re
import time
from typing import Optional

import requests

from fetch_markets import parse_json_field

log = logging.getLogger(__name__)

GAMMA_URL   = "https://gamma-api.polymarket.com"
MAX_RETRIES = 3
TIMEOUT     = 15

CONDITION_ID = re.compile(r"^0x[0-9a-fA-F]{64}$")
# polymarket.com/event/<event>[/<market>] or /market/<market>, with an optional locale prefix
POLYMARKET_URL = re.compile(
    r"polymarket\.com/(?:[a-z]{2}/)?(?:(?P<kind>event|market)/)(?P<first>[^/?#]+)(?:/(?P<second>[^/?#]+))?"
)
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")


class GammaError(Exception):
    """Gamma could not be reached (as opposed to a market not existing)."""


def _get(path: str, params: dict) -> object:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(f"{GAMMA_URL}{path}", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise GammaError(f"{path}: {e}") from e
            time.sleep(attempt)


def market_by_condition_id(condition_id: str) -> Optional[dict]:
    """The market with this condition ID, open or closed."""
    for closed in ("false", "true"):
        found = _get("/markets", {"condition_ids": condition_id, "closed": closed})
        if found:
            return found[0]
    return None


def market_by_slug(slug: str) -> Optional[dict]:
    found = _get("/markets", {"slug": slug})
    return found[0] if found else None


def event_markets(slug: str) -> list[dict]:
    found = _get("/events", {"slug": slug})
    return found[0].get("markets", []) if found else []


def search(text: str, limit: int = 10) -> list[dict]:
    """Open markets of the active events matching `text`."""
    found = _get("/public-search", {"q": text, "limit_per_type": limit,
                                    "events_status": "active", "keep_closed_markets": 0})
    return [m for e in found.get("events", []) for m in e.get("markets", []) if not m.get("closed")]


def resolve(q: str) -> list[dict]:
    """Markets for a query: a polymarket.com URL, a condition ID, a slug, or free text."""
    q = q.strip()
    if CONDITION_ID.match(q):
        m = market_by_condition_id(q)
        return [m] if m else []
    url = POLYMARKET_URL.search(q)
    if url:
        if url["kind"] == "event" and not url["second"]:
            return event_markets(url["first"])
        m = market_by_slug(url["second"] or url["first"])
        return [m] if m else []
    if SLUG.match(q):
        m = market_by_slug(q)
        if m:
            return [m]
        markets = event_markets(q)
        if markets:
            return markets
    return search(q)


def yes_price(m: dict) -> Optional[float]:
    """The first outcome's price (the YES token, as in training)."""
    prices = parse_json_field(m.get("outcomePrices"), [])
    try:
        return float(prices[0])
    except (IndexError, TypeError, ValueError):
        return None


def is_binary(m: dict) -> bool:
    return len(parse_json_field(m.get("outcomes"), [])) == 2


def summary(m: dict) -> dict:
    return {
        "market_id": m.get("conditionId"),
        "question":  m.get("question"),
        "slug":      m.get("slug"),
        "end_date":  m.get("endDate"),
        "price":     yes_price(m),
        "binary":    is_binary(m),
        "closed":    bool(m.get("closed")),
    }
