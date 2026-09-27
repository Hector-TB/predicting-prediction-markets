"""
Categorize Markets with Claude API
=====================================
Reads questions from polymarket_markets_meta.csv, sends them in batches
to the Claude API for classification, then writes the category back to
both polymarket_markets_meta.csv and polymarket_ml_dataset.csv.

Categories:
    politics_us      - US elections, congress, presidency, domestic policy
    politics_global  - Foreign elections, heads of state, international politics
    crypto           - Bitcoin, ETH, DeFi, tokens, NFTs, blockchain
    sports           - NFL, NBA, MLB, NHL, soccer, tennis, any sport
    finance          - Fed, inflation, stocks, IPOs, economic indicators
    geopolitics      - Wars, sanctions, treaties, military conflicts
    science_tech     - AI, space, FDA approvals, climate, technology
    entertainment    - Celebrity, TV, music, film, pop culture
    other            - Anything that doesn't fit above

Run after fetch_markets.py and build_snapshots.py.
"""

import anthropic
import pandas as pd
from dotenv import load_dotenv

from stream_parquet import rewrite
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

META_CSV            = DATA_DIR / "polymarket_markets_meta.csv"
DATASET_PARQUET     = DATA_DIR / "polymarket_ml_dataset.parquet"

BATCH_SIZE          = 100       # questions per API call
SLEEP_BETWEEN_CALLS = 0.5       # seconds between batches
MAX_RETRIES         = 3

VALID_CATEGORIES = {
    "politics_us",
    "politics_global",
    "crypto",
    "sports",
    "finance",
    "geopolitics",
    "science_tech",
    "entertainment",
    "other",
}

SYSTEM_PROMPT = """You are a classification assistant. You will be given a list of prediction market questions and must classify each one into exactly one of these categories:

- politics_us: US elections, congress, presidency, supreme court, domestic US policy
- politics_global: Foreign elections, heads of state, international political events
- crypto: Bitcoin, Ethereum, DeFi, tokens, NFTs, blockchain, crypto exchanges
- sports: Any sport — NFL, NBA, MLB, NHL, soccer, tennis, golf, MMA, esports
- finance: Fed, interest rates, inflation, stocks, IPOs, GDP, economic indicators
- geopolitics: Wars, military conflicts, sanctions, treaties, territorial disputes
- science_tech: AI, space, NASA, FDA approvals, climate, scientific discoveries, tech products
- entertainment: Celebrity, TV shows, music, film, awards, pop culture, social media
- other: Anything that doesn't clearly fit the above

You will receive a JSON array of objects, each with an integer "id" and a "question".
Return exactly one result per input object, echoing its "id"."""

# Structured output: every answer is keyed by the input's id and the category is
# constrained to VALID_CATEGORIES, so answers can't shift onto the wrong question.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":       {"type": "integer"},
                    "category": {"type": "string", "enum": sorted(VALID_CATEGORIES)},
                },
                "required": ["id", "category"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}


# ─────────────────────────────────────────────
# CLASSIFY BATCH
# ─────────────────────────────────────────────

class FatalAPIError(Exception):
    """An error no retry can fix (billing, auth) — stop the run and save progress."""


def _is_fatal(e: Exception) -> bool:
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return True
    # Out of credits comes back as a 400 invalid_request_error
    return isinstance(e, anthropic.BadRequestError) and "credit balance" in str(e).lower()


def classify_batch(client: anthropic.Anthropic,
                   batch: list[dict]) -> Optional[list[Optional[str]]]:
    """
    Send a batch of questions to Claude.
    Returns a list aligned with batch: a category per question, or None for any
    question the response didn't cover. Returns None if the whole batch failed.
    """
    payload = json.dumps([{"id": i, "question": m["question"]} for i, m in enumerate(batch)])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8192,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": payload}],
                output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            )
            if message.stop_reason != "end_turn":
                raise ValueError(f"stop_reason={message.stop_reason}")
            text = next(b.text for b in message.content if b.type == "text")
            by_id = {r["id"]: r["category"] for r in json.loads(text)["results"]}
            cats = [by_id.get(i) for i in range(len(batch))]
            n_missing = sum(c is None for c in cats)
            if n_missing:
                print(f"({n_missing} missing — left for next run) ", end="")
            return cats

        except Exception as e:
            if _is_fatal(e):
                raise FatalAPIError(str(e)) from e
            wait = 2 ** attempt
            if attempt < MAX_RETRIES:
                print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"    All attempts failed for batch: {e}")
                return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  POLYMARKET — LLM CATEGORY ENRICHMENT")
    print("█" * 60 + "\n")

    # ── Load meta CSV ─────────────────────────
    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found.")
        return

    meta = pd.read_csv(META_CSV)
    print(f"Loaded {len(meta):,} markets from {META_CSV}")

    # ── Find uncategorised markets ─────────────
    # Re-classify rows that are missing, "other", or a legacy Gamma label
    # outside VALID_CATEGORIES (e.g. "US-current-affairs")
    cat = meta["category"].fillna("").astype(str).str.strip().str.lower()
    needs_category = meta[~cat.isin(VALID_CATEGORIES) | (cat == "other")].copy()

    already_done = len(meta) - len(needs_category)
    print(f"  Already categorised: {already_done:,}")
    print(f"  Need categorisation: {len(needs_category):,}")

    stopped_early = False
    if needs_category.empty:
        print("\nAll markets already categorised.")
    else:
        client = anthropic.Anthropic()

        # Build batches
        records = [
            {"id": str(row["market_id"]), "question": str(row["question"])}
            for _, row in needs_category.iterrows()
        ]

        n_batches    = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        all_categories = [None] * len(records)  # positional; None = batch failed, retried next run

        print(f"\nClassifying in {n_batches} batches of {BATCH_SIZE}...\n")

        for b in range(n_batches):
            start  = b * BATCH_SIZE
            batch  = records[start : start + BATCH_SIZE]
            pct    = (b + 1) / n_batches * 100
            print(f"  [{pct:5.1f}%] Batch {b+1}/{n_batches} ({len(batch)} markets)...",
                  end=" ", flush=True)

            try:
                categories = classify_batch(client, batch)
            except FatalAPIError as e:
                stopped_early = True
                print(f"\n\n  STOPPING — {e}")
                print("  Saving what was classified so far; re-run to continue.")
                break

            if categories:
                all_categories[start : start + len(batch)] = categories
                print(f"ok — {sum(c is not None for c in categories)} classified")
            else:
                print("failed — left uncategorised for next run")

            time.sleep(SLEEP_BETWEEN_CALLS)

        # ── Apply to meta ──────────────────────
        id_to_category = {r["id"]: c for r, c in zip(records, all_categories) if c is not None}
        new_cat = meta["market_id"].astype(str).map(id_to_category)
        meta["category"] = new_cat.fillna(meta["category"])
        n_failed = sum(c is None for c in all_categories)
        if n_failed:
            print(f"\n  WARNING: {n_failed:,} markets in failed batches kept their old category")

    # ── Print category distribution ───────────
    print(f"\nCategory distribution:")
    dist = meta["category"].value_counts()
    for cat, count in dist.items():
        pct = count / len(meta) * 100
        print(f"  {cat:<20} {count:>6,}  ({pct:5.1f}%)")

    # ── Save meta CSV ─────────────────────────
    meta.to_csv(META_CSV, index=False)
    print(f"\nSaved updated categories to {META_CSV}")

    # ── Update dataset parquet ────────────────
    if not DATASET_PARQUET.exists():
        print(f"\n{DATASET_PARQUET} not found — skipping dataset update.")
        sys.exit(1 if stopped_early else 0)

    print(f"\nUpdating categories in {DATASET_PARQUET}...")

    # Build market_id -> category mapping from updated meta
    # Keep NaN as NaN (astype(str) would turn it into the string "nan")
    cat_map = dict(zip(meta["market_id"].astype(str), meta["category"]))

    def stamp(df: pd.DataFrame) -> pd.DataFrame:
        df["category"] = df["market_id"].astype(str).map(cat_map).fillna("other")
        return df

    # Streamed in batches: the full dataset doesn't fit in memory on 8 GB machines
    _, rows = rewrite(DATASET_PARQUET, DATASET_PARQUET, stamp)
    print(f"Saved updated categories to {DATASET_PARQUET} ({rows:,} rows)")

    print(f"\n{'=' * 60}")
    print(f"  {'STOPPED EARLY — re-run to finish' if stopped_early else 'COMPLETE'}")
    print(f"{'=' * 60}")
    # Non-zero exit so run_pipeline.py reports the step as incomplete
    if stopped_early:
        sys.exit(1)


if __name__ == "__main__":
    main()