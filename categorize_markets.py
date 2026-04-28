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
import json
import time
import os
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

META_CSV            = "polymarket_markets_meta.csv"
DATASET_CSV         = "polymarket_ml_dataset.csv"

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

You will receive a JSON array of objects with "id" and "question" fields.
Respond with ONLY a JSON array of objects with "id" and "category" fields.
No preamble, no explanation, no markdown — just the raw JSON array."""


# ─────────────────────────────────────────────
# CLASSIFY BATCH
# ─────────────────────────────────────────────

def classify_batch(client: anthropic.Anthropic,
                   batch: list[dict]) -> Optional[dict[str, str]]:
    """
    Send a batch of {id, question} dicts to Claude.
    Returns a dict mapping market_id -> category.
    """
    payload = json.dumps([{"id": m["id"], "question": m["question"]} for m in batch])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": payload}],
            )
            raw = message.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            results = json.loads(raw)

            # Validate and build mapping
            mapping = {}
            for r in results:
                mid = str(r.get("id", ""))
                cat = r.get("category", "other").strip().lower()
                if cat not in VALID_CATEGORIES:
                    cat = "other"
                mapping[mid] = cat

            return mapping

        except Exception as e:
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
    if not os.path.exists(META_CSV):
        print(f"ERROR: {META_CSV} not found.")
        return

    meta = pd.read_csv(META_CSV)
    print(f"Loaded {len(meta):,} markets from {META_CSV}")

    # ── Find uncategorised markets ─────────────
    # Only re-classify rows where category is missing or "other"
    needs_category = meta[
        meta["category"].isna() |
        (meta["category"].str.strip().str.lower() == "other") |
        (meta["category"].str.strip() == "")
    ].copy()

    already_done = len(meta) - len(needs_category)
    print(f"  Already categorised: {already_done:,}")
    print(f"  Need categorisation: {len(needs_category):,}")

    if needs_category.empty:
        print("\nAll markets already categorised.")
    else:
        client = anthropic.Anthropic()

        # Build batches
        records = [
            {"id": str(row["market_id"]), "question": str(row["question"])}
            for _, row in needs_category.iterrows()
        ]

        n_batches   = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        all_mapping = {}

        print(f"\nClassifying in {n_batches} batches of {BATCH_SIZE}...\n")

        for b in range(n_batches):
            batch  = records[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            pct    = (b + 1) / n_batches * 100
            print(f"  [{pct:5.1f}%] Batch {b+1}/{n_batches} ({len(batch)} markets)...",
                  end=" ", flush=True)

            mapping = classify_batch(client, batch)

            if mapping:
                all_mapping.update(mapping)
                # Count category distribution in this batch
                cats = list(mapping.values())
                print(f"ok — {len(mapping)} classified")
            else:
                # Fallback: assign "other" for failed batches
                for r in batch:
                    all_mapping[r["id"]] = "other"
                print("failed — assigned 'other'")

            time.sleep(SLEEP_BETWEEN_CALLS)

        # ── Apply to meta ──────────────────────
        meta["category"] = meta.apply(
            lambda row: all_mapping.get(str(row["market_id"]), row["category"])
            if pd.isna(row["category"]) or str(row["category"]).strip().lower() in ("other", "")
            else row["category"],
            axis=1
        )

    # ── Print category distribution ───────────
    print(f"\nCategory distribution:")
    dist = meta["category"].value_counts()
    for cat, count in dist.items():
        pct = count / len(meta) * 100
        print(f"  {cat:<20} {count:>6,}  ({pct:5.1f}%)")

    # ── Save meta CSV ─────────────────────────
    meta.to_csv(META_CSV, index=False)
    print(f"\nSaved updated categories to {META_CSV}")

    # ── Update dataset CSV ────────────────────
    if not os.path.exists(DATASET_CSV):
        print(f"\n{DATASET_CSV} not found — skipping dataset update.")
        return

    print(f"\nUpdating categories in {DATASET_CSV}...")

    # Build market_id -> category mapping from updated meta
    cat_map = dict(zip(meta["market_id"].astype(str),
                       meta["category"].astype(str)))

    # Read dataset in chunks to handle large files
    chunk_size  = 100_000
    chunks      = []
    first_write = True
    tmp_path    = DATASET_CSV + ".tmp"

    reader = pd.read_csv(DATASET_CSV, chunksize=chunk_size)
    total_rows = 0

    for chunk in reader:
        chunk["category"] = chunk["market_id"].astype(str).map(cat_map).fillna("other")
        chunk.to_csv(tmp_path, mode="a", header=first_write, index=False)
        first_write  = False
        total_rows  += len(chunk)
        print(f"  Processed {total_rows:,} rows...", end="\r")

    # Replace original with updated
    os.replace(tmp_path, DATASET_CSV)
    print(f"\nSaved updated categories to {DATASET_CSV} ({total_rows:,} rows)")

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()