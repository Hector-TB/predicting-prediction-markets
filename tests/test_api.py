"""Offline tests of the API's feature fills, warnings (ADR-025) and query parsing."""

import numpy as np
import pandas as pd
import pytest

import api  # noqa: F401  (sets up the import paths)
from api import gamma
from api.predict import Market, PredictError, build_features, parse_market
from api.releases import ServingRelease

NOW  = pd.Timestamp("2026-10-09 12:00", tz="UTC")
FILL = {"days_before_close": 86.6, "pct_lifetime_elapsed": 0.4376, "duration_days": 172.0}
RELEASE = ServingRelease("rX", {"dataset_version": "vX", "models": {}}, FILL)


def market(**kw) -> Market:
    base = dict(market_id="0x" + "a" * 64, question="Q?", slug="q", event_slug="e", token="1",
                start=NOW - pd.Timedelta(days=60), end=NOW + pd.Timedelta(days=60), end_raw="x",
                volume=50_000.0, price=0.4)
    return Market(**{**base, **kw})


def prices(n: int, days: float = 60, price: float = 0.4) -> pd.DataFrame:
    ts = pd.date_range(end=NOW, periods=n, freq=pd.Timedelta(days=days) / max(n - 1, 1))
    return pd.DataFrame({"timestamp": ts, "price": np.full(n, price)})


def codes(m: Market, p) -> list[str]:
    return build_features(m, p, NOW, RELEASE)[1]


def test_ordinary_market_has_no_warnings():
    feats, c = build_features(market(), prices(120), NOW, RELEASE)
    assert c == []
    assert feats["days_before_close"] == 60.0 and feats["pct_lifetime_elapsed"] == 0.5


def test_no_end_date_uses_release_fill():
    feats, c = build_features(market(end=None), prices(120), NOW, RELEASE)
    assert "no_end_date" in c
    assert {k: feats[k] for k in FILL} == FILL


def test_past_end_date_is_zero_days_left():
    feats, c = build_features(market(end=NOW - pd.Timedelta(days=3)), prices(120), NOW, RELEASE)
    assert "past_end_date" in c and "near_close" not in c
    assert feats["days_before_close"] == 0.0 and feats["pct_lifetime_elapsed"] == 1.0


def test_no_history_uses_gamma_price_with_flat_windows():
    feats, c = build_features(market(price=0.3), None, NOW, RELEASE)
    assert "few_prices" in c
    assert feats["price_at_snapshot"] == 0.3 and feats["price_mean_7d"] == 0.3
    assert feats["price_volatility_7d"] == 0.0


def test_no_history_and_no_price_is_an_error():
    with pytest.raises(PredictError) as e:
        build_features(market(price=None), None, NOW, RELEASE)
    assert e.value.code == "no_price"


@pytest.mark.parametrize("kw, p, code", [
    ({}, prices(5), "few_prices"),
    ({"start": NOW - pd.Timedelta(days=3), "end": NOW + pd.Timedelta(days=60)}, prices(30, days=3), "young_market"),
    ({}, prices(120, price=0.97), "settled_price"),
    ({}, prices(120, price=0.05), "settled_price"),
    ({"end": NOW + pd.Timedelta(days=10)}, prices(120), "near_close"),
    ({"start": NOW - pd.Timedelta(days=300), "end": NOW + pd.Timedelta(days=100)}, prices(120), "long_market"),
    ({"start": NOW - pd.Timedelta(days=15), "end": NOW + pd.Timedelta(days=10)}, prices(30, days=15), "outside_filters"),
    ({"volume": 500.0}, prices(120), "outside_filters"),
])
def test_warning(kw, p, code):
    assert code in codes(market(**kw), p)


def test_parse_market_rejects_non_binary_and_closed():
    m = {"conditionId": "0x1", "outcomes": '["A", "B", "C"]', "createdAt": "2026-01-01T00:00:00Z"}
    with pytest.raises(PredictError, match="YES/NO"):
        parse_market(m)
    with pytest.raises(PredictError) as e:
        parse_market({**m, "outcomes": '["Yes", "No"]', "closed": True})
    assert e.value.code == "closed"


def test_parse_market_end_not_after_start_counts_as_missing():
    m = parse_market({"conditionId": "0x1", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.2", "0.8"]',
                      "createdAt": "2026-01-01T00:00:00Z", "endDate": "2025-12-01T00:00:00Z",
                      "clobTokenIds": '["11", "22"]'})
    assert m.end is None and m.end_raw == "2025-12-01T00:00:00Z" and m.token == "11" and m.price == 0.2


@pytest.mark.parametrize("q, expected", [
    ("0x" + "b" * 64,                                         ("/markets", "condition_ids")),
    ("https://polymarket.com/event/fed-cuts",                 ("/events", "slug")),
    ("https://polymarket.com/event/fed-cuts/one-cut",         ("/markets", "slug")),
    ("polymarket.com/market/one-cut?tid=1",                   ("/markets", "slug")),
    ("fed rate cut",                                          ("/public-search", "q")),
])
def test_resolve_routes_queries(monkeypatch, q, expected):
    calls = []

    def fake_get(path, params):
        calls.append((path, params))
        return {"events": []} if path == "/public-search" else []
    monkeypatch.setattr(gamma, "_get", fake_get)
    gamma.resolve(q)
    path, key = expected
    assert calls[0][0] == path and key in calls[0][1]
