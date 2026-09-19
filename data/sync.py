"""
data/sync.py
============
Push/pull data files to/from S3.

Training scripts always read local files — run `pull` on a new machine before training.
Size-based comparison skips files that are already up to date.

Usage:
    python data/sync.py status           # compare local vs S3
    python data/sync.py push             # upload local files to S3
    python data/sync.py pull             # download files from S3
    python data/sync.py push --dry-run   # preview without uploading
    python data/sync.py pull --dry-run   # preview without downloading
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

BUCKET    = os.getenv("S3_BUCKET")
REGION    = os.getenv("AWS_REGION", "us-east-1")
S3_PREFIX = os.getenv("S3_DATA_PREFIX", "data")

# All data files managed by this script
DATA_FILES = [
    # canonical datasets (what training scripts use)
    "polymarket_ml_dataset_clean.parquet",
    "polymarket_ml_dataset_with_trends_clean.parquet",
    # intermediate / raw (useful to keep but not required for training)
    "polymarket_ml_dataset.parquet",
    "polymarket_ml_dataset_with_trends_part1.parquet",
    "polymarket_ml_dataset_with_trends_part2.parquet",
    "polymarket_ml_dataset_with_trends_part1_clean.parquet",
    "polymarket_ml_dataset_with_trends_part2_clean.parquet",
    # small feature tables
    "category_trends_features.parquet",
    # market metadata (gitignored CSV)
    "polymarket_markets_meta.csv",
]

SEP = "─" * 88


def make_client():
    if not BUCKET:
        print("ERROR: S3_BUCKET not set in .env")
        sys.exit(1)
    try:
        return boto3.client(
            "s3",
            region_name=REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except NoCredentialsError:
        print("ERROR: AWS credentials missing. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")
        sys.exit(1)


def s3_key(fname: str) -> str:
    return f"{S3_PREFIX}/{fname}"


def local_size(fname: str) -> int | None:
    p = DATA_DIR / fname
    return p.stat().st_size if p.exists() else None


def remote_size(s3, fname: str) -> int | None:
    try:
        return s3.head_object(Bucket=BUCKET, Key=s3_key(fname))["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def fmt_mb(n: int | None) -> str:
    return f"{n / 1e6:>7.1f} MB" if n is not None else "       —"


# ── commands ──────────────────────────────────────────────────────────────────

def status(s3):
    print(f"\n  {'File':<50} {'Local':>10}  {'S3':>10}  Status")
    print(f"  {SEP}")
    for fname in DATA_FILES:
        loc = local_size(fname)
        rem = remote_size(s3, fname)
        if loc and rem:
            state = "in sync" if loc == rem else "SIZE MISMATCH"
        elif loc:
            state = "local only"
        elif rem:
            state = "S3 only"
        else:
            state = "missing"
        print(f"  {fname:<50} {fmt_mb(loc)}  {fmt_mb(rem)}  {state}")
    print()


def push(s3, dry_run: bool):
    print(f"\nPushing to s3://{BUCKET}/{S3_PREFIX}/\n")
    for fname in DATA_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"  skip   {fname}  (not found locally)")
            continue
        loc = path.stat().st_size
        rem = remote_size(s3, fname)
        if rem == loc:
            print(f"  skip   {fname}  ({fmt_mb(loc).strip()} — already up to date)")
            continue
        verb = "upload" if not dry_run else "would upload"
        print(f"  {verb}  {fname}  ({fmt_mb(loc).strip()}) ...", end="", flush=True)
        if not dry_run:
            s3.upload_file(str(path), BUCKET, s3_key(fname))
            print("  ✓")
        else:
            print()
    suffix = "" if not dry_run else "  (dry run — nothing uploaded)"
    print(f"\nDone.{suffix}\n")


def pull(s3, dry_run: bool):
    print(f"\nPulling from s3://{BUCKET}/{S3_PREFIX}/\n")
    for fname in DATA_FILES:
        rem = remote_size(s3, fname)
        if rem is None:
            print(f"  skip   {fname}  (not in S3)")
            continue
        loc = local_size(fname)
        if loc == rem:
            print(f"  skip   {fname}  ({fmt_mb(rem).strip()} — already up to date)")
            continue
        verb = "download" if not dry_run else "would download"
        print(f"  {verb}  {fname}  ({fmt_mb(rem).strip()}) ...", end="", flush=True)
        if not dry_run:
            s3.download_file(BUCKET, s3_key(fname), str(DATA_DIR / fname))
            print("  ✓")
        else:
            print()
    suffix = "" if not dry_run else "  (dry run — nothing downloaded)"
    print(f"\nDone.{suffix}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sync data files with S3")
    parser.add_argument("command", choices=["push", "pull", "status"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without doing it")
    args = parser.parse_args()

    s3 = make_client()
    {"push": push, "pull": pull, "status": status}[args.command](
        *([s3, args.dry_run] if args.command != "status" else [s3])
    )


if __name__ == "__main__":
    main()
