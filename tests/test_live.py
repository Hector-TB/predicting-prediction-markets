"""Offline tests of the live track record: resolution rule, daily sample, scoring (ADR-026)."""

import numpy as np
import pandas as pd
import pytest
from resolve_lookups import resolution
from sample_markets import REJECT, draw_order, eligible

import api  # noqa: F401  (sets up the import paths)
from api import predict as predict_mod
from api import track_record
from api.predict import PredictError, predict_raw
from api.releases import ServingRelease

NOW = pd.Timestamp("2026-10-12 12:00", tz="UTC")
RELEASE = ServingRelease("rX", {"dataset_version": "vX", "models": {}}, {})


def gamma_market(**kw) -> dict:
    base = {"conditionId": "0x" + "a" * 64, "question": "Q?", "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["1", "2"]', "closed": False, "outcomePrices": '["0.4", "0.6"]',
            "createdAt": "2026-08-01T00:00:00Z", "startDate": "2026-08-01T00:00:00Z",
            "endDate": "2026-12-31T00:00:00Z", "volumeNum": 50_000}
    return {**base, **kw}


# ── resolution (ADR-004) ─────────────────────────────────────────────────────

@pytest.mark.parametrize("prices, expected", [
    ('["1", "0"]', ("resolved", 1)), ('["0.96", "0.04"]', ("resolved", 1)),
    ('["0", "1"]', ("resolved", 0)), ('["0.05", "0.95"]', ("resolved", 0)),
    ('["0.5", "0.5"]', ("ambiguous", None)), (None, ("ambiguous", None)),
])
def test_resolution_rule(prices, expected):
    assert resolution({"outcomePrices": prices}) == expected


# ── eligibility and draw ─────────────────────────────────────────────────────

def test_ordinary_open_market_is_eligible():
    assert eligible(gamma_market(), NOW)


@pytest.mark.parametrize("kw", [
    {"outcomes": '["A", "B", "C"]'}, {"clobTokenIds": "[]"}, {"closed": True},
    {"volumeNum": 999}, {"endDate": "2026-10-01T00:00:00Z"}, {"endDate": None},
    {"createdAt": "2026-10-05T00:00:00Z"},                                   # inside the 14-day burn-in
    {"createdAt": "2026-09-01T00:00:00Z", "endDate": "2026-09-30T23:00:00Z"},  # under 30 days long
    {"startDate": "2022-12-31T00:00:00Z"},
    {"outcomePrices": '["0.95", "0.05"]'}, {"outcomePrices": '["0.05", "0.95"]'},
])
def test_ineligible(kw):
    assert not eligible(gamma_market(**kw), NOW)


def test_draw_order_depends_only_on_seed_and_set():
    ms = [gamma_market(conditionId=f"0x{i:064x}") for i in range(50)]
    a = draw_order(ms, "2026-10-12")
    assert a == draw_order(list(reversed(ms)), "2026-10-12")
    assert a != draw_order(ms, "2026-10-13")
    assert sorted(m["conditionId"] for m in a) == sorted(m["conditionId"] for m in ms)


def test_predict_raw_rejects_before_the_category_call(monkeypatch):
    monkeypatch.setattr(predict_mod.features, "fetch_price_history", lambda *a, **k: None)
    monkeypatch.setattr(predict_mod, "category", lambda m: pytest.fail("category called"))
    with pytest.raises(PredictError) as e:
        predict_raw(gamma_market(), RELEASE, NOW, reject=REJECT)   # no history -> few_prices
    assert e.value.code == "ineligible"


# ── scoring ──────────────────────────────────────────────────────────────────

MODELS = ["logistic_regression", "gradient_boosting", "random_forest"]


def lookups(n: int, source: str = "sample", seed: int = 0, release: str = "r2") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    price = np.clip(0.5 * y + rng.uniform(0.05, 0.5, n), 0.06, 0.94)
    df = pd.DataFrame({"source": source, "market_id": [f"{source}{seed}-{i}" for i in range(n)],
                       "release": release, "market_price": price, "resolution_status": "resolved",
                       "outcome": y.astype(float), "duration_days": rng.integers(30, 700, n).astype(float)})
    for m in MODELS:
        df[m] = np.clip(price + rng.normal(0, 0.05, n), 0.01, 0.99)
    return df


def test_scores_only_from_min_resolved():
    few = track_record.score(lookups(track_record.MIN_RESOLVED - 1), {})
    live = few["releases"][0]["live"]["sample"]
    assert live["resolved"] == track_record.MIN_RESOLVED - 1 and live["metrics"] is None

    many = track_record.score(lookups(150), {})["releases"][0]["live"]
    m = many["sample"]["metrics"]
    assert m["n"] == 150 and set(m["models"]) == set(MODELS)
    assert {"auc", "log_loss", "brier"} <= set(m["models"]["gradient_boosting"]["vs_market"])
    assert many["sample"]["metrics_length_weighted"]["n"] == 150
    assert "metrics_length_weighted" not in many["search"] and many["search"]["markets"] == 0


def test_open_and_ambiguous_are_counted_not_scored():
    df = lookups(150)
    df.loc[:19, ["resolution_status", "outcome"]] = ["open", np.nan]
    df.loc[20:29, ["resolution_status", "outcome"]] = ["ambiguous", np.nan]
    live = track_record.score(df, {})["releases"][0]["live"]["sample"]
    assert (live["open"], live["ambiguous"], live["resolved"]) == (20, 10, 120)
    assert live["metrics"]["n"] == 120


def test_sources_and_releases_are_kept_apart():
    df = pd.concat([lookups(120), lookups(30, source="search", seed=1), lookups(5, release="r3", seed=2)])
    out = track_record.score(df, {"r1": {"gradient_boosting": {"auc": 0.87}}})
    assert [r["release"] for r in out["releases"]] == ["r1", "r2", "r3"]
    r1, r2, _ = out["releases"]
    assert r1["live"]["sample"]["markets"] == 0 and r1["backtest_per_market"]["gradient_boosting"]["auc"] == 0.87
    assert r2["live"]["sample"]["markets"] == 120 and r2["live"]["search"]["markets"] == 30


def test_length_weights_undo_long_market_over_representation():
    df = pd.DataFrame({"duration_days": [30.0, 380.0, 10.0]})
    assert list(track_record.eligible_days(df)) == [16.0, 366.0, 1.0]
