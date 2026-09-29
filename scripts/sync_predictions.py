"""
scripts/sync_predictions.py
===========================
Test-set prediction files live on S3, not in git (interim, until model releases
exist — ADR-019 amendment):

    s3://<bucket>/predictions/<dataset version>/<model>/<file>.csv

The dataset version comes from the file's `dataset_version` column. Files from
before that column existed (the course paper's v1 runs) need --legacy-version.
Each object carries its sha256, row count, git commit and upload time as
metadata. Re-pushing a changed file replaces it; S3 bucket versioning keeps
the old copy.

Usage:
    python scripts/sync_predictions.py list
    python scripts/sync_predictions.py push                          # every local prediction file
    python scripts/sync_predictions.py push --model logistic_regression
    python scripts/sync_predictions.py push --legacy-version v1      # files without dataset_version
    python scripts/sync_predictions.py pull --version v3 [--force]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "data"))
from sync import BUCKET, exists, git_info, make_client, sha256_file  # noqa: E402

log = logging.getLogger(__name__)

PREFIX = "predictions"


def local_files(model: str | None) -> list[Path]:
    pattern = f"{model}/predictions/*.csv" if model else "*/predictions/*.csv"
    return sorted(MODELS_DIR.glob(pattern))


def file_version(path: Path, legacy_version: str | None) -> str | None:
    cols = pd.read_csv(path, nrows=0).columns
    if "dataset_version" not in cols:
        return legacy_version
    versions = pd.read_csv(path, usecols=["dataset_version"])["dataset_version"].astype(str).unique()
    if len(versions) != 1:
        raise ValueError(f"{path} mixes dataset versions: {sorted(versions)}")
    return versions[0]


def s3_key(version: str, path: Path) -> str:
    model = path.parent.parent.name
    return f"{PREFIX}/{version}/{model}/{path.name}"


def cmd_push(s3, args) -> None:
    files = local_files(args.model)
    if not files:
        sys.exit("No prediction files found.")
    commit = git_info()["git_commit"]
    for path in files:
        version = file_version(path, args.legacy_version)
        rel = path.relative_to(ROOT)
        if version is None:
            log.warning("  skip %s — no dataset_version column (pass --legacy-version)", rel)
            continue
        k = s3_key(version, path)
        sha = sha256_file(path)
        if exists(s3, k):
            old = s3.head_object(Bucket=BUCKET, Key=k)["Metadata"].get("sha256")
            if old == sha:
                log.info("  same    %s → s3://%s/%s", rel, BUCKET, k)
                continue
            log.info("  replace %s → s3://%s/%s  (old copy kept by bucket versioning)", rel, BUCKET, k)
        else:
            log.info("  upload  %s → s3://%s/%s", rel, BUCKET, k)
        rows = sum(1 for _ in open(path)) - 1
        s3.upload_file(str(path), BUCKET, k, ExtraArgs={"Metadata": {
            "sha256": sha, "rows": str(rows), "git_commit": commit,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }})
        size = s3.head_object(Bucket=BUCKET, Key=k)["ContentLength"]
        if size != path.stat().st_size:
            sys.exit(f"ERROR: size mismatch after upload of {rel}")


def cmd_list(s3, args) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/"):
        for o in page.get("Contents", []):
            log.info("  %-70s %8.1f MB  %s", o["Key"], o["Size"] / 1e6, o["LastModified"].strftime("%Y-%m-%d %H:%M"))


def cmd_pull(s3, args) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/{args.version}/")
            for o in page.get("Contents", [])]
    if not keys:
        sys.exit(f"No predictions on S3 for dataset {args.version}.")
    for k in keys:
        _, _, model, fname = k.split("/", 3)
        dest = MODELS_DIR / model / "predictions" / fname
        sha = s3.head_object(Bucket=BUCKET, Key=k)["Metadata"].get("sha256")
        if dest.exists():
            if sha and sha256_file(dest) == sha:
                log.info("  same    %s", dest.relative_to(ROOT))
                continue
            if not args.force:
                log.warning("  skip    %s — local file differs (use --force)", dest.relative_to(ROOT))
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, k, str(dest))
        if sha and sha256_file(dest) != sha:
            sys.exit(f"ERROR: checksum mismatch for {dest}")
        log.info("  pulled  %s", dest.relative_to(ROOT))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--model", help="one model directory, e.g. logistic_regression")
    p.add_argument("--legacy-version", help="dataset version for files without a dataset_version column")
    sub.add_parser("list")
    p = sub.add_parser("pull")
    p.add_argument("--version", required=True)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()

    s3 = make_client()
    {"push": cmd_push, "list": cmd_list, "pull": cmd_pull}[args.cmd](s3, args)


if __name__ == "__main__":
    main()
