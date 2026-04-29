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
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"

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

You will receive a JSON array of question strings.
Respond with ONLY a JSON array of category strings, in the same order.
No preamble, no explanation, no markdown — just the raw JSON array."""


# ─────────────────────────────────────────────
# CLASSIFY BATCH
# ─────────────────────────────────────────────

def classify_batch(client: anthropic.Anthropic,
                   batch: list[dict]) -> Optional[list[str]]:
    """
    Send a batch of questions to Claude.
    Returns a list of category strings in the same order as batch.
    """
    payload = json.dumps([m["question"] for m in batch])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": payload}],
            )
            raw = message.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            results = json.loads(raw)
            cats = [
                r.strip().lower() if r.strip().lower() in VALID_CATEGORIES else "other"
                for r in results
            ]
            # Truncate if too long, pad with "other" if too short
            if len(cats) != len(batch):
                print(f"(count mismatch: got {len(cats)}, expected {len(batch)} — adjusting) ",
                      end="")
                cats = (cats + ["other"] * len(batch))[:len(batch)]
            return cats

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
    if not META_CSV.exists():
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

        n_batches    = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        all_categories = ["other"] * len(records)  # positional, same order as records

        print(f"\nClassifying in {n_batches} batches of {BATCH_SIZE}...\n")

        for b in range(n_batches):
            start  = b * BATCH_SIZE
            batch  = records[start : start + BATCH_SIZE]
            pct    = (b + 1) / n_batches * 100
            print(f"  [{pct:5.1f}%] Batch {b+1}/{n_batches} ({len(batch)} markets)...",
                  end=" ", flush=True)

            categories = classify_batch(client, batch)

            if categories:
                all_categories[start : start + len(batch)] = categories
                print(f"ok — {len(categories)} classified")
            else:
                print("failed — assigned 'other'")

            time.sleep(SLEEP_BETWEEN_CALLS)

        # ── Apply to meta ──────────────────────
        id_to_category = {r["id"]: cat for r, cat in zip(records, all_categories)}
        meta["category"] = meta.apply(
            lambda row: id_to_category.get(str(row["market_id"]), row["category"])
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

    # ── Update dataset parquet ────────────────
    if not DATASET_PARQUET.exists():
        print(f"\n{DATASET_PARQUET} not found — skipping dataset update.")
        return

    print(f"\nUpdating categories in {DATASET_PARQUET}...")

    # Build market_id -> category mapping from updated meta
    cat_map = dict(zip(meta["market_id"].astype(str),
                       meta["category"].astype(str)))

    df = pd.read_parquet(DATASET_PARQUET)
    df["category"] = df["market_id"].astype(str).map(cat_map).fillna("other")
    tmp_path = DATASET_PARQUET.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(DATASET_PARQUET)
    print(f"Saved updated categories to {DATASET_PARQUET} ({len(df):,} rows)")

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()