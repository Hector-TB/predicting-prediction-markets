"""
scripts/stage_backup.py
=======================
Back up an unpublished build to s3://<bucket>/staging/<name>/ — insurance until
it is reviewed and published with `data/sync.py publish` (ADR-016). Not a
dataset version: staging copies may be deleted once the version is published.

Optionally waits for running jobs (pipeline, coverage check) to exit first.
Every upload is verified by size against S3.

Usage:
    python scripts/stage_backup.py --name 2026-09-28-v3 [--wait-pid 123 --wait-pid 456]
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DATA_FILES = [
    "polymarket_ml_dataset.csv",                        # raw snapshot build
    "polymarket_ml_dataset.csv.build.json",             # its build settings
    "polymarket_ml_dataset.parquet",
    "polymarket_ml_dataset_with_trends.parquet",
    "polymarket_ml_dataset_clean.parquet",              # training data
    "polymarket_ml_dataset_with_trends_clean.parquet",
    "polymarket_markets_meta.csv",
    "market_closed_times.csv",
    "category_trends_raw.csv",
    "category_trends_features.parquet",
]
LOG_PATTERNS = ["pipeline_*.log", "coverage_*.csv", "coverage_run_*.log", "build_snapshots_*.log"]

log = logging.getLogger(__name__)


def running(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0


def main(name: str, wait_pids: list[int]) -> int:
    load_dotenv(ROOT / ".env")
    for pid in wait_pids:
        if running(pid):
            log.info("waiting for PID %d ...", pid)
        while running(pid):
            time.sleep(30)
    if wait_pids:
        log.info("all waited-for jobs have exited")

    files = [DATA / f for f in DATA_FILES if (DATA / f).exists()]
    files += sorted((DATA / "fetch_cache").glob("*.csv"))
    for pattern in LOG_PATTERNS:
        files += sorted((ROOT / "logs").glob(pattern))

    csv = DATA / "polymarket_ml_dataset.csv"
    if csv.exists():
        with open(csv, "rb") as f:
            f.seek(-1, 2)
            log.info("snapshot CSV %.0f MB, ends on a complete row: %s",
                     csv.stat().st_size / 1e6, f.read(1) == b"\n")

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    prefix = f"staging/{name}"
    bad = 0
    for p in files:
        key = f"{prefix}/{p.relative_to(ROOT)}"
        s3.upload_file(str(p), bucket, key)
        remote = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        ok = remote == p.stat().st_size
        bad += not ok
        log.info("uploaded %-62s %9.1f MB  %s", p.relative_to(ROOT), p.stat().st_size / 1e6,
                 "OK" if ok else f"SIZE MISMATCH ({remote} bytes on S3)")
    log.info("%s — %d files in s3://%s/%s/", "DONE" if not bad else f"DONE WITH {bad} MISMATCHES",
             len(files), bucket, prefix)
    return bad


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("botocore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Back up an unpublished build to S3 staging")
    parser.add_argument("--name", required=True, help="staging folder, e.g. 2026-09-28-v3")
    parser.add_argument("--wait-pid", type=int, action="append", default=[],
                        help="wait for this process to exit first (repeatable)")
    args = parser.parse_args()
    sys.exit(1 if main(args.name, args.wait_pid) else 0)
