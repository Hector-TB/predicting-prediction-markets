"""
Store every /predict lookup (ADR-017; migrations 005 + 006)
==========================================================
One transaction per lookup: upsert the market (split NULL, resolution_status
'open'), insert a `snapshots` row with the full feature vector and warnings,
and one `predictions` row per model, linked to the release's `model_runs`.
"""

import json
import logging
import os

import psycopg2

from api.predict import Market
from api.releases import ServingRelease
from release import DB_NAMES   # scripts/release.py: model → model_runs.model_name

log = logging.getLogger(__name__)

# Typed snapshot columns from migration 001 (the full vector is in `features`)
TYPED = ["price_at_snapshot", "pct_lifetime_elapsed", "days_before_close",
         "price_mean_7d", "price_volatility_7d", "price_change_7d", "price_trend_7d",
         "price_mean_14d", "price_volatility_14d", "price_change_14d", "price_trend_14d"]


class StoreError(Exception):
    pass


def connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise StoreError("DATABASE_URL not set")
    return psycopg2.connect(url, connect_timeout=5)


def model_run_ids(cur, release: ServingRelease) -> dict[str, int]:
    cur.execute("SELECT model_name, id FROM model_runs WHERE release = %s AND dataset_tag = %s",
                (release.name, f"clean_{release.dataset_version}"))
    by_db_name = dict(cur.fetchall())
    ids = {m: by_db_name.get(DB_NAMES[m]) for m in release.models}
    missing = [m for m, i in ids.items() if i is None]
    if missing:
        raise StoreError(f"no model_runs rows for {release.name} ({', '.join(missing)}): "
                         f"run python scripts/release.py sync-db {release.name}")
    return ids


def save_lookup(result: dict, market: Market, release: ServingRelease, source: str = "search") -> int:
    """Store one lookup; returns the snapshot id. `source`: 'search' (a user) or 'sample' (ADR-026)."""
    feats = result["features"]
    with connect() as conn, conn.cursor() as cur:
        runs = model_run_ids(cur, release)
        # Dataset markets (split set) are left alone; looked-up markets track Gamma's latest values
        cur.execute(
            """INSERT INTO markets (market_id, clob_token_id, question, category, slug, start_date,
                                    end_date, duration_days, resolution_status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open')
               ON CONFLICT (market_id) DO UPDATE SET
                   question = EXCLUDED.question, slug = EXCLUDED.slug, end_date = EXCLUDED.end_date,
                   duration_days = EXCLUDED.duration_days,
                   category = COALESCE(markets.category, EXCLUDED.category)
               WHERE markets.split IS NULL""",
            (market.market_id, market.token, market.question, result["market"]["category"],
             market.slug, market.start, market.end, market.duration_days))
        cur.execute(
            f"""INSERT INTO snapshots (market_id, snapshot_timestamp, {", ".join(TYPED)}, features,
                                       warnings, market_price, end_date_seen, volume_to_date, release, source)
                VALUES (%s, %s, {", ".join(["%s"] * len(TYPED))}, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                RETURNING id""",
            (market.market_id, result["as_of"], *[feats[c] for c in TYPED], json.dumps(feats),
             json.dumps([w["code"] for w in result["warnings"]]), market.price, market.end_raw,
             market.volume, release.name, source))
        snapshot_id = cur.fetchone()[0]
        for p in result["predictions"]:
            cur.execute("INSERT INTO predictions (snapshot_id, model_run_id, probability) VALUES (%s, %s, %s)",
                        (snapshot_id, runs[p["model"]], p["probability"]))
    conn.close()
    return snapshot_id
