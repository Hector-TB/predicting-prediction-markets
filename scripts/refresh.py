"""
scripts/refresh.py
==================
One-command refresh (ADR-019): new markets → dataset vN+1 → (later stages)
candidate release. Stages, each recorded in logs/refresh_<run id>/state.json:

  0 preflight   clean git; local data = the LATEST published version; meta checks;
                a production release exists
  1 fetch       incremental fetch by scheduled end date (ADR-020); stops if nothing is new
  2 split       move T: 80/20 by resolution time over all markets (ADR-021)
  3 build       retire the previous build's raw CSV, then snapshots for new markets,
                merge, categories, Trends join, leakage filter (run_pipeline.py)
  4 gates       meta checks; coverage (recent window, or full if the last full recount
                is > 30 days old); check_snapshots --parent; smoke test
  5 publish     sync.py publish vN+1 --parent vN; commit data/manifests/vN+1.json

Stages 6–8 (train, candidate release, change report) come next (ROADMAP step 4).

Usage:
    python scripts/refresh.py                  # new run
    python scripts/refresh.py --resume         # continue the newest unfinished run
    python scripts/refresh.py --stop-after 0   # run up to and including a stage
    python scripts/refresh.py --no-publish     # stop before publishing
    python scripts/refresh.py --full-coverage  # full coverage recount regardless of age
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
PIPELINE = ROOT / "data_collection_pipeline"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(PIPELINE))

from fetch_markets import last_fetch_time, last_full_coverage  # noqa: E402
from fix_leakage import split_cutoff  # noqa: E402
from sync import (  # noqa: E402
    BUCKET, DATASETS_PREFIX, exists, git_info, make_client, read_text, sha256_file,
)

log = logging.getLogger(__name__)

STAGES = ["preflight", "fetch", "split", "build", "gates", "publish"]
COVERAGE_LOOKBACK_DAYS = 90     # ADR-019: 3× the fetch's 30-day look-back
FULL_COVERAGE_EVERY_DAYS = 30
META = DATA / "polymarket_markets_meta.csv"
RAW_CSV = DATA / "polymarket_ml_dataset.csv"
RAW_PQ = DATA / "polymarket_ml_dataset.parquet"
CLOSED = DATA / "market_closed_times.csv"
VERIFY_FILES = ["polymarket_markets_meta.csv", "polymarket_ml_dataset.parquet",
                "polymarket_ml_dataset_clean.parquet", "polymarket_ml_dataset_with_trends_clean.parquet"]


class StageFailed(Exception):
    pass


# ─────────────────────────────────────────────
# RUN STATE
# ─────────────────────────────────────────────

class Run:
    def __init__(self, path: Path):
        self.dir = path
        self.file = path / "state.json"
        self.state = json.loads(self.file.read_text())

    @classmethod
    def new(cls) -> "Run":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        d = LOGS / f"refresh_{run_id}"
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps({"run_id": run_id, "started": now(), "stages": {}}, indent=2))
        return cls(d)

    @classmethod
    def latest_unfinished(cls) -> "Run":
        for d in sorted(LOGS.glob("refresh_*"), reverse=True):
            run = cls(d)
            if not run.state.get("finished"):
                return run
        raise SystemExit("No unfinished refresh to resume.")

    def save(self) -> None:
        self.file.write_text(json.dumps(self.state, indent=2, default=str) + "\n")

    def done(self, stage: str) -> bool:
        return self.state["stages"].get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, **info) -> None:
        self.state["stages"].setdefault(stage, {}).update(status=status, at=now(), **info)
        self.save()

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, value):
        self.state[key] = value
        self.save()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_step(run: Run, name: str, cmd: list[str]) -> None:
    """Run a script, logging its output to the run folder; raise StageFailed on a non-zero exit."""
    logfile = run.dir / f"{name}.log"
    log.info("  → %s  (log: %s)", " ".join(cmd), logfile.relative_to(ROOT))
    t0 = time.time()
    with logfile.open("a") as f:
        rc = subprocess.run([sys.executable, *cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log.info("    %s in %.0f min", "ok" if rc == 0 else f"FAILED (exit {rc})", (time.time() - t0) / 60)
    if rc != 0:
        tail = logfile.read_text().splitlines()[-15:]
        raise StageFailed(f"{name} failed:\n    " + "\n    ".join(tail))


def meta_ids() -> set:
    return set(pd.read_csv(META, usecols=["market_id"])["market_id"])


def current_cutoff() -> pd.Timestamp:
    closed = pd.read_csv(CLOSED)
    closed["closed_time"] = pd.to_datetime(closed["closed_time"], utc=True, format="mixed")
    return split_cutoff(closed)


# ─────────────────────────────────────────────
# STAGES
# ─────────────────────────────────────────────

def stage_preflight(run: Run, args) -> None:
    if git_info()["uncommitted_changes"]:
        raise StageFailed("uncommitted changes — commit first, so the new version records exact code (ADR-016).")
    manifest = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else None
    if manifest is None:
        raise StageFailed("local data has no manifest — pull the latest dataset version first.")
    s3 = make_client()
    latest = read_text(s3, f"{DATASETS_PREFIX}/LATEST").strip()
    if manifest["version"] != latest:
        raise StageFailed(f"local data is {manifest['version']}, but LATEST is {latest} — pull it first.")
    for f in VERIFY_FILES:
        if sha256_file(DATA / f) != manifest["files"][f]["sha256"]:
            raise StageFailed(f"{f} differs from {latest}'s manifest — local data was modified.")
    if not exists(s3, "models/PRODUCTION"):
        raise StageFailed("no production release — run scripts/release.py init first (ADR-019).")
    run_step(run, "meta_checks", [str(PIPELINE / "meta_checks.py")])

    parent = manifest["version"]
    run["parent"] = parent
    run["next_version"] = f"v{int(parent[1:]) + 1}"
    run["prev_fetch"] = str(last_fetch_time())
    run["markets_before"] = len(meta_ids())
    run["cutoff_before"] = str(current_cutoff())
    log.info("  %s is LATEST and intact; production is %s; last fetch %s",
             parent, read_text(s3, "models/PRODUCTION").strip(), run["prev_fetch"])


def stage_fetch(run: Run, args) -> None:
    run_step(run, "fetch", [str(PIPELINE / "fetch_markets.py")])
    added = len(meta_ids()) - run["markets_before"]
    run["markets_added"] = added
    log.info("  %s new markets", f"{added:,}")
    if added <= 0:
        run["finished"] = now()
        raise SystemExit("Nothing new since the last fetch — refresh finished, nothing published.")


def stage_split(run: Run, args) -> None:
    run_step(run, "recompute_split", [str(ROOT / "scripts" / "recompute_split.py")])
    before, after = pd.Timestamp(run["cutoff_before"]), current_cutoff()
    run["cutoff_after"] = str(after)
    log.info("  T: %s → %s", before, after)
    if after < before:
        raise StageFailed(f"T would move backwards ({before} → {after}) — investigate before building.")


def retire_raw_csv(run: Run) -> None:
    """The previous build's CSV must already be published in the parent's raw parquet (ADR-019)."""
    if run.state.get("csv_retired") or not RAW_CSV.exists():
        return
    in_parquet = set(pq.read_table(RAW_PQ, columns=["market_id"]).column("market_id").to_pylist())
    in_csv = set()
    for chunk in pd.read_csv(RAW_CSV, usecols=["market_id"], chunksize=500_000):
        in_csv.update(chunk["market_id"].unique())
    missing = in_csv - in_parquet
    if missing:
        raise StageFailed(f"{len(missing):,} markets in {RAW_CSV.name} aren't in the published raw parquet "
                          "— that build was never published; don't delete it.")
    log.info("  Retiring %s: all %s markets are in %s's raw parquet", RAW_CSV.name, f"{len(in_csv):,}", run["parent"])
    RAW_CSV.unlink()
    (DATA / "polymarket_ml_dataset.csv.build.json").unlink(missing_ok=True)
    run["csv_retired"] = True


def stage_build(run: Run, args) -> None:
    retire_raw_csv(run)
    run_step(run, "pipeline", [str(PIPELINE / "run_pipeline.py"), "--skip-markets", "--skip-trends"])


def stage_gates(run: Run, args) -> None:
    run_step(run, "meta_checks", [str(PIPELINE / "meta_checks.py")])

    last_full = last_full_coverage()
    due = last_full is None or (pd.Timestamp.now(tz="UTC") - last_full).days > FULL_COVERAGE_EVERY_DAYS
    if args.full_coverage or due:
        log.info("  Coverage: full recount (last full: %s)", last_full)
        run_step(run, "coverage", [str(ROOT / "scripts" / "coverage_check.py")])
        run["coverage"] = "full"
    else:
        end_from = (pd.Timestamp(run["prev_fetch"]) - pd.Timedelta(days=COVERAGE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        log.info("  Coverage: scheduled end >= %s (last full recount %s)", end_from, last_full.date())
        run_step(run, "coverage", [str(ROOT / "scripts" / "coverage_check.py"), "--end-from", end_from])
        run["coverage"] = f"recent window from {end_from}"

    run_step(run, "check_snapshots", [str(ROOT / "scripts" / "check_snapshots.py"), "--parent", run["parent"]])
    run_step(run, "smoke_test", [str(ROOT / "scripts" / "smoke_test.py")])


def stage_publish(run: Run, args) -> None:
    if args.no_publish:
        raise SystemExit("--no-publish: stopping before publish. Resume with: python scripts/refresh.py --resume")
    version, parent = run["next_version"], run["parent"]
    notes = (f"Refresh {run['run_id']} of {parent}: +{run['markets_added']:,} markets (incremental fetch by end "
             f"date, ADR-020); T {run['cutoff_before'][:16]} → {run['cutoff_after'][:16]} (ADR-021). "
             f"Gates passed: meta checks, coverage ({run['coverage']}), check_snapshots --parent {parent}, "
             f"smoke test. Trends as in {parent}.")
    run_step(run, "publish", [str(DATA / "sync.py"), "publish", version, "--parent", parent, "--notes", notes])
    manifest = DATA / "manifests" / f"{version}.json"
    subprocess.run(["git", "add", str(manifest)], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"Publish dataset {version} manifest (refresh {run['run_id']})",
                    "--", str(manifest)], cwd=ROOT, check=True)
    log.info("  Published %s; manifest committed", version)


RUNNERS = dict(zip(STAGES, [stage_preflight, stage_fetch, stage_split, stage_build, stage_gates, stage_publish]))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", action="store_true", help="continue the newest unfinished run")
    parser.add_argument("--stop-after", type=int, choices=range(len(STAGES)), help="last stage to run")
    parser.add_argument("--no-publish", action="store_true", help="stop before stage 5")
    parser.add_argument("--full-coverage", action="store_true", help="full coverage recount regardless of age")
    args = parser.parse_args()

    if not args.resume:
        unfinished = [d for d in LOGS.glob("refresh_*") if not Run(d).state.get("finished")]
        if unfinished:
            raise SystemExit(f"Unfinished refresh {unfinished[-1].name} — resume it with --resume "
                             "(or mark it finished in its state.json).")
    run = Run.latest_unfinished() if args.resume else Run.new()
    log.info("Refresh %s", run["run_id"])

    last = len(STAGES) - 1 if args.stop_after is None else args.stop_after
    for i, stage in enumerate(STAGES[: last + 1]):
        if run.done(stage):
            log.info("\n[%d] %s — done earlier", i, stage)
            continue
        log.info("\n[%d] %s", i, stage)
        run.mark(stage, "running")
        try:
            RUNNERS[stage](run, args)
        except StageFailed as e:
            run.mark(stage, "failed", error=str(e))
            log.error("\nStage %s failed: %s\nFix it, then: python scripts/refresh.py --resume", stage, e)
            sys.exit(1)
        run.mark(stage, "done")

    if last == len(STAGES) - 1:
        run["finished"] = now()
        log.info("\nRefresh %s complete: %s published.", run["run_id"], run["next_version"])
    else:
        log.info("\nStopped after stage %d (%s). Continue with --resume.", last, STAGES[last])


if __name__ == "__main__":
    main()
