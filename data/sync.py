"""
data/sync.py
============
Versioned datasets on S3 (ADR-016).

Each dataset version is immutable and lives in its own prefix with a manifest:

    s3://<bucket>/datasets/v1/<files> + manifest.json
    s3://<bucket>/datasets/v2/...
    s3://<bucket>/datasets/LATEST        ← name of the newest version

The manifest records what the version covers (market dates, fetch date), how it
was built (git commit, ADRs), row/market counts, and a SHA-256 for every file.
A copy of each manifest is also written to data/manifests/<version>.json, which
is tracked in git, and data/manifest.json records which version is checked out
locally.

Usage:
    python data/sync.py list                        # versions on S3
    python data/sync.py status                      # which version is local, and is it intact?
    python data/sync.py pull                        # download LATEST
    python data/sync.py pull --version v1           # download a specific version
    python data/sync.py publish v2 --parent v1 --notes "…"   # upload local data as a NEW version
    python data/sync.py freeze-legacy v1 --notes "…"         # one-time: adopt the old flat data/ prefix

publish refuses to overwrite an existing version; pull refuses to overwrite
local files that differ from the target version unless --force is given.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"        # tracked in git
LOCAL_MANIFEST = DATA_DIR / "manifest.json"   # which version is checked out (gitignored)

load_dotenv(ROOT / ".env")

BUCKET          = os.getenv("S3_BUCKET")
REGION          = os.getenv("AWS_REGION", "us-east-1")
LEGACY_PREFIX   = os.getenv("S3_DATA_PREFIX", "data")          # pre-ADR-016 flat layout
DATASETS_PREFIX = os.getenv("S3_DATASETS_PREFIX", "datasets")

META_FILE  = "polymarket_markets_meta.csv"
CLEAN_FILE = "polymarket_ml_dataset_clean.parquet"

# What a published version contains
REQUIRED_FILES = [
    META_FILE,
    CLEAN_FILE,                                          # training data (base features)
    "polymarket_ml_dataset_with_trends_clean.parquet",   # training data (+ Google Trends)
]
OPTIONAL_FILES = [
    "polymarket_ml_dataset.parquet",                     # all snapshots, before the leakage filter
    "polymarket_ml_dataset_with_trends.parquet",
    "category_trends_features.parquet",
    "category_trends_raw.csv",
    "market_closed_times.csv",                           # closedTime cache used by the leakage filter
]

VERSION_RE = re.compile(r"^v\d+$")


# ─────────────────────────────────────────────
# S3 + HASH HELPERS
# ─────────────────────────────────────────────

def make_client():
    if not BUCKET:
        sys.exit("ERROR: S3_BUCKET not set in .env")
    return boto3.client("s3", region_name=REGION)


def key(version: str, fname: str) -> str:
    return f"{DATASETS_PREFIX}/{version}/{fname}"


def exists(s3, k: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=k)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def read_text(s3, k: str) -> str:
    return s3.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_s3(s3, k: str) -> str:
    h = hashlib.sha256()
    for chunk in s3.get_object(Bucket=BUCKET, Key=k)["Body"].iter_chunks(1 << 20):
        h.update(chunk)
    return h.hexdigest()


def parquet_rows(source) -> int | None:
    try:
        return pq.ParquetFile(source).metadata.num_rows
    except Exception:
        return None


# ─────────────────────────────────────────────
# MANIFEST CONTENTS
# ─────────────────────────────────────────────

def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    # Uncommitted changes to code (data files are gitignored and don't count)
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith("data/manifest")]
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(dirty)}


def adrs_in_effect() -> list[str]:
    return sorted(p.stem[:3] for p in (ROOT / "docs" / "decisions").glob("[0-9][0-9][0-9]-*.md") if p.stem[:3] != "000")


def summarize(meta_src, clean_src) -> dict:
    """Coverage and counts from the meta CSV and the clean parquet (file paths or buffers)."""
    meta = pd.read_csv(meta_src, usecols=lambda c: c in {"market_id", "start_date", "end_date", "split", "category"})
    start = pd.to_datetime(meta["start_date"], format="ISO8601", utc=True)
    out = {
        "markets_fetched": int(meta["market_id"].nunique()),
        "market_start_date_min": str(start.min().date()),
        "market_start_date_max": str(start.max().date()),
    }
    if "split" in meta:
        out["markets_fetched_by_split"] = {k: int(v) for k, v in meta["split"].value_counts().items()}

    markets: dict[str, set] = {}
    rows: dict[str, int] = {}
    yes = n = 0
    ts_min = ts_max = None
    split_ts: dict[str, list] = {}   # split -> [first, last] snapshot
    for b in pq.ParquetFile(clean_src).iter_batches(batch_size=250_000,
                                                    columns=["market_id", "split", "outcome", "snapshot_timestamp"]):
        df = b.to_pandas()
        ts = pd.to_datetime(df["snapshot_timestamp"], format="mixed", utc=True)
        ts_min = ts.min() if ts_min is None else min(ts_min, ts.min())
        ts_max = ts.max() if ts_max is None else max(ts_max, ts.max())
        for split, g in df.groupby("split"):
            markets.setdefault(split, set()).update(g["market_id"].unique())
            rows[split] = rows.get(split, 0) + len(g)
            lo, hi = ts[g.index].min(), ts[g.index].max()
            prev = split_ts.get(split)
            split_ts[split] = [lo, hi] if prev is None else [min(prev[0], lo), max(prev[1], hi)]
        yes += int(df["outcome"].sum())
        n += len(df)
    out.update({
        "clean_snapshots": n,
        "clean_markets": len(set().union(*markets.values())) if markets else 0,
        "clean_snapshots_by_split": rows,
        "clean_markets_by_split": {k: len(v) for k, v in markets.items()},
        "clean_snapshot_yes_rate": round(yes / n, 4) if n else None,
        "snapshot_timestamp_min": str(ts_min),
        "snapshot_timestamp_max": str(ts_max),
        # ADR-021: the last train snapshot must precede the first test snapshot
        "snapshot_range_by_split": {k: [str(a), str(b)] for k, (a, b) in split_ts.items()},
    })
    return out


def write_manifest_copies(manifest: dict) -> None:
    MANIFESTS_DIR.mkdir(exist_ok=True)
    text = json.dumps(manifest, indent=2) + "\n"
    (MANIFESTS_DIR / f"{manifest['version']}.json").write_text(text)
    LOCAL_MANIFEST.write_text(text)


def set_latest(s3, version: str) -> None:
    current = read_text(s3, f"{DATASETS_PREFIX}/LATEST").strip() if exists(s3, f"{DATASETS_PREFIX}/LATEST") else None
    if current is None or int(version[1:]) > int(current[1:]):
        s3.put_object(Bucket=BUCKET, Key=f"{DATASETS_PREFIX}/LATEST", Body=version.encode())
        log.info("  LATEST → %s", version)


def check_new_version(s3, version: str) -> None:
    if not VERSION_RE.match(version):
        sys.exit(f"ERROR: version must look like v1, v2, … (got {version!r})")
    if exists(s3, key(version, "manifest.json")):
        sys.exit(f"ERROR: {version} already exists on S3 — versions are immutable; publish a new version instead")


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

def cmd_list(s3, args) -> None:
    latest = read_text(s3, f"{DATASETS_PREFIX}/LATEST").strip() if exists(s3, f"{DATASETS_PREFIX}/LATEST") else None
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{DATASETS_PREFIX}/", Delimiter="/")
    versions = sorted((p["Prefix"].split("/")[1] for p in resp.get("CommonPrefixes", [])),
                      key=lambda v: int(v[1:]) if VERSION_RE.match(v) else -1)
    if not versions:
        log.info("No dataset versions on s3://%s/%s/", BUCKET, DATASETS_PREFIX)
    for v in versions:
        if not exists(s3, key(v, "manifest.json")):
            log.info("  %-4s (incomplete — no manifest)", v)
            continue
        m = json.loads(read_text(s3, key(v, "manifest.json")))
        c = m.get("contents", {})
        log.info("  %-4s%s  created %s  fetched %s  %s markets, %s clean snapshots — %s",
                 v, " (LATEST)" if v == latest else "         ", m["created_at"][:10], m.get("fetched_on", "?"),
                 f"{c.get('clean_markets', 0):,}", f"{c.get('clean_snapshots', 0):,}", m.get("notes", ""))


def cmd_status(s3, args) -> None:
    if not LOCAL_MANIFEST.exists():
        log.info("Local data has no manifest — it isn't a checked-out version (e.g. an unpublished rebuild).")
        return
    m = json.loads(LOCAL_MANIFEST.read_text())
    log.info("Local data is %s (created %s)", m["version"], m["created_at"][:10])
    for fname, info in m["files"].items():
        p = DATA_DIR / fname
        if not p.exists():
            state = "missing"
        elif p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
            state = "MODIFIED since pull/publish"
        else:
            state = "ok"
        log.info("  %-52s %s", fname, state)


def cmd_publish(s3, args) -> None:
    version = args.version
    check_new_version(s3, version)
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        sys.exit(f"ERROR: required files missing locally: {missing}")
    files = [f for f in REQUIRED_FILES + OPTIONAL_FILES if (DATA_DIR / f).exists()]

    # ADR-023: never publish meta with estimated or placeholder values
    sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
    from meta_checks import assert_meta_ok
    assert_meta_ok(pd.read_csv(DATA_DIR / META_FILE, dtype={"clob_token_id": str}))

    log.info("Building manifest for %s ...", version)
    manifest = {
        "version":    version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent":     args.parent,
        "notes":      args.notes or "",
        "fetched_on": args.fetched_on or datetime.fromtimestamp(
            (DATA_DIR / META_FILE).stat().st_mtime, timezone.utc).date().isoformat(),
        "code":       {**git_info(), "adrs": adrs_in_effect()},
        "contents":   summarize(DATA_DIR / META_FILE, DATA_DIR / CLEAN_FILE),
        "files":      {},
    }
    if manifest["code"]["uncommitted_changes"]:
        log.warning("  WARNING: the working tree has uncommitted changes — the git commit won't fully describe the code")
    for f in files:
        p = DATA_DIR / f
        manifest["files"][f] = {"bytes": p.stat().st_size, "sha256": sha256_file(p), "rows": parquet_rows(p)}

    log.info(json.dumps(manifest["contents"], indent=2))
    if args.dry_run:
        log.info("Dry run — would upload %d files to s3://%s/%s/%s/", len(files), BUCKET, DATASETS_PREFIX, version)
        return

    for f in files:
        log.info("  upload %-52s %8.1f MB", f, manifest["files"][f]["bytes"] / 1e6)
        s3.upload_file(str(DATA_DIR / f), BUCKET, key(version, f))
    # The manifest goes last: its presence marks the version as complete
    s3.put_object(Bucket=BUCKET, Key=key(version, "manifest.json"), Body=json.dumps(manifest, indent=2).encode())
    set_latest(s3, version)
    write_manifest_copies(manifest)
    log.info("Published %s. Commit data/manifests/%s.json to record it in git.", version, version)


def cmd_pull(s3, args) -> None:
    version = args.version or read_text(s3, f"{DATASETS_PREFIX}/LATEST").strip()
    if not exists(s3, key(version, "manifest.json")):
        sys.exit(f"ERROR: no complete version {version} on S3")
    manifest = json.loads(read_text(s3, key(version, "manifest.json")))

    todo, conflicts = [], []
    for fname, info in manifest["files"].items():
        p = DATA_DIR / fname
        if p.exists() and p.stat().st_size == info["bytes"] and sha256_file(p) == info["sha256"]:
            continue
        (conflicts if p.exists() else todo).append(fname)
    if conflicts and not args.force:
        log.error("These local files differ from %s and would be overwritten:", version)
        for f in conflicts:
            log.error("  %s", f)
        sys.exit("Refusing to overwrite. Publish your local data first, or re-run with --force.")

    for fname in todo + conflicts:
        info = manifest["files"][fname]
        log.info("  %s %-52s %8.1f MB", "would download" if args.dry_run else "download", fname, info["bytes"] / 1e6)
        if args.dry_run:
            continue
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, delete=False) as tmp:
            s3.download_fileobj(BUCKET, key(version, fname), tmp)
        if sha256_file(Path(tmp.name)) != info["sha256"]:
            os.unlink(tmp.name)
            sys.exit(f"ERROR: checksum mismatch for {fname} — download discarded")
        os.replace(tmp.name, DATA_DIR / fname)
    if not args.dry_run:
        write_manifest_copies(manifest)
        log.info("Local data is now %s.", version)


def cmd_freeze_legacy(s3, args) -> None:
    """One-time: copy the pre-ADR-016 flat prefix into an immutable version, server-side."""
    version = args.version
    check_new_version(s3, version)
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{LEGACY_PREFIX}/").get("Contents", [])
    if not objs:
        sys.exit(f"ERROR: nothing under s3://{BUCKET}/{LEGACY_PREFIX}/")

    files = {}
    for o in objs:
        fname = o["Key"].split("/", 1)[1]
        log.info("  %s %-52s %8.1f MB", "would copy" if args.dry_run else "copy", fname, o["Size"] / 1e6)
        if not args.dry_run:
            s3.copy({"Bucket": BUCKET, "Key": o["Key"]}, BUCKET, key(version, fname))
        body = s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read() if fname.endswith(".parquet") else None
        files[fname] = {"bytes": o["Size"], "sha256": sha256_s3(s3, o["Key"]),
                        "rows": parquet_rows(io.BytesIO(body)) if body else None,
                        "uploaded_to_legacy_prefix": o["LastModified"].isoformat()}
    if args.dry_run:
        return

    def fetch(fname):
        return io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{LEGACY_PREFIX}/{fname}")["Body"].read())

    manifest = {
        "version":    version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent":     None,
        "notes":      args.notes or "",
        "fetched_on": args.fetched_on,
        "code":       {"git_commit": None, "adrs": args.adrs.split(",") if args.adrs else [],
                       "provenance": f"adopted from the pre-ADR-016 s3://{BUCKET}/{LEGACY_PREFIX}/ prefix"},
        "contents":   summarize(fetch(META_FILE), fetch(CLEAN_FILE)),
        "files":      files,
    }
    s3.put_object(Bucket=BUCKET, Key=key(version, "manifest.json"), Body=json.dumps(manifest, indent=2).encode())
    set_latest(s3, version)
    MANIFESTS_DIR.mkdir(exist_ok=True)
    (MANIFESTS_DIR / f"{version}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info(json.dumps(manifest["contents"], indent=2))
    log.info("Froze %s. The legacy prefix was left in place.", version)


def main():
    parser = argparse.ArgumentParser(description="Versioned datasets on S3 (ADR-016)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    p = sub.add_parser("pull")
    p.add_argument("--version", help="default: LATEST")
    p.add_argument("--force", action="store_true", help="overwrite local files that differ")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("publish")
    p.add_argument("version")
    p.add_argument("--parent", help="version this one was derived from, e.g. v1")
    p.add_argument("--notes", help="what changed and why")
    p.add_argument("--fetched-on", help="date the market list was fetched (default: meta CSV mtime)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("freeze-legacy")
    p.add_argument("version")
    p.add_argument("--notes")
    p.add_argument("--fetched-on")
    p.add_argument("--adrs", help="comma-separated ADR numbers the data was built under")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("push")  # removed — kept only to explain what replaced it
    args = parser.parse_args()

    if args.command == "push":
        sys.exit("`push` was replaced by `publish <version>` (ADR-016): versions are immutable.")
    s3 = make_client()
    {"list": cmd_list, "status": cmd_status, "pull": cmd_pull, "publish": cmd_publish,
     "freeze-legacy": cmd_freeze_legacy}[args.command](s3, args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("botocore").setLevel(logging.WARNING)
    main()
