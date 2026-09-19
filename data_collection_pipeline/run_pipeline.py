"""
data_collection_pipeline/run_pipeline.py
=========================================
Runs the full data refresh pipeline end-to-end.

Steps:
  1. fetch_markets.py       — new resolved markets from Gamma API (incremental)
  2. build_snapshots.py     — CLOB price history for new markets (checkpoint/resume)
  3. fix_dataset.py         — clean + merge into parquet (skipped if no new CSV rows)
  4. categorize_markets.py  — LLM category for new markets (skipped if nothing new)
  5. fetch_category_trends.py — extend Google Trends to today (incremental)
  6. build_trend_features.py  — recompute rolling trend features
  7. merge_trends.py          — merge snapshots + trends
  8. fix_leakage.py           — remove post-resolution snapshots

Usage:
    python data_collection_pipeline/run_pipeline.py
    python data_collection_pipeline/run_pipeline.py --skip-snapshots
    python data_collection_pipeline/run_pipeline.py --trends-only
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
PIPELINE   = Path(__file__).parent
DATA_DIR   = ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEP = "─" * 60


def run(script: Path, label: str) -> bool:
    log.info("\n%s\n  STEP: %s\n%s", SEP, label, SEP)
    t0     = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log.error("  FAILED (exit %d) — %.1fs", result.returncode, elapsed)
        return False
    log.info("  OK — %.1fs", elapsed)
    return True


def has_new_csv_data() -> bool:
    csv = DATA_DIR / "polymarket_ml_dataset.csv"
    return csv.exists() and csv.stat().st_size > 0


def main():
    parser = argparse.ArgumentParser(description="Run the full data refresh pipeline")
    parser.add_argument("--skip-markets",   action="store_true", help="Skip fetch_markets.py")
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip build_snapshots.py")
    parser.add_argument("--skip-trends",    action="store_true", help="Skip Trends fetch + features")
    parser.add_argument("--trends-only",    action="store_true", help="Only run Trends steps (5-8)")
    args = parser.parse_args()

    log.info("\n%s", "█" * 60)
    log.info("  POLYMARKET — FULL DATA PIPELINE")
    log.info("%s\n", "█" * 60)

    steps_ok = True

    # ── Market data ──────────────────────────────────────────────────────────
    if not args.trends_only:
        if not args.skip_markets:
            if not run(PIPELINE / "fetch_markets.py", "fetch_markets — Gamma API"):
                log.error("Aborting: fetch_markets failed.")
                sys.exit(1)

        if not args.skip_snapshots:
            if not run(PIPELINE / "build_snapshots.py", "build_snapshots — CLOB price history"):
                log.error("Aborting: build_snapshots failed.")
                sys.exit(1)

        if has_new_csv_data():
            if not run(PIPELINE / "fix_dataset.py", "fix_dataset — clean + write parquet"):
                steps_ok = False
            if not run(PIPELINE / "categorize_markets.py", "categorize_markets — LLM categories"):
                steps_ok = False
        else:
            log.info("\n%s\n  Skipping fix_dataset + categorize (no new snapshot rows)\n%s", SEP, SEP)

    # ── Trends ───────────────────────────────────────────────────────────────
    if not args.skip_trends:
        if not run(PIPELINE / "fetch_category_trends.py", "fetch_category_trends — Google Trends"):
            steps_ok = False
        if not run(PIPELINE / "build_trend_features.py", "build_trend_features — rolling features"):
            steps_ok = False
        if not run(PIPELINE / "merge_trends.py", "merge_trends — snapshots + trends"):
            steps_ok = False
        if not run(PIPELINE / "fix_leakage.py", "fix_leakage — remove post-resolution rows"):
            steps_ok = False

    log.info("\n%s", "=" * 60)
    if steps_ok:
        log.info("  PIPELINE COMPLETE")
        log.info("  Next: python data/sync.py push")
    else:
        log.info("  PIPELINE COMPLETE WITH ERRORS — check output above")
    log.info("%s\n", "=" * 60)

    sys.exit(0 if steps_ok else 1)


if __name__ == "__main__":
    main()
