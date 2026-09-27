# ADR-016: Immutable, Manifest-Described Dataset Versions on S3

**Date:** 2026-09-27  
**Status:** Accepted (supersedes the flat S3 layout in ADR-012)  
**Deciders:** Hector Thompson Baroni  
**Code location:** `data/sync.py`, `data/manifests/`, `models/common/evaluation.py` (`dataset_version`), `db/load_parquet.py`

---

## Context

ADR-012 put the data files in one flat S3 prefix (`data/`) and `sync.py push` overwrote them in place. Bucket versioning had never been enabled. The v2 rebuild (ADR-013/014) would therefore have permanently replaced v1, the only copy of the data behind the course paper, on its first push.

Checking S3 before freezing it (2026-09-27) showed why a version needs more than a date:

- The clean files on S3 hold 21,316 markets / 1,510,332 snapshots, a superset of the paper's 20,948 / 1,448,142. 368 markets were added by an incremental fetch on the day of upload, and nothing recorded it.
- `polymarket_markets_meta.csv` on S3 was a stray 210-market test file, not the metadata for those snapshots.
- v2 isn't v1 plus newer rows. The re-fetch found markets v1 had missed, even from 2025, and the leakage rules changed, so the same end date can produce different data.

## Decision

1. **Every published dataset is an immutable version** under `s3://<bucket>/datasets/<version>/`, named `v1`, `v2`, … `datasets/LATEST` names the newest.
2. **Each version has a `manifest.json`**, uploaded last so its presence marks the version complete. It contains:
   - coverage: market start-date range, fetch date, snapshot timestamp range;
   - code: git commit, branch, uncommitted-changes flag, ADRs in effect;
   - contents: markets and snapshots overall and by split, where the test split starts, YES rate;
   - every file's size, row count and SHA-256;
   - parent version and free-text notes.
3. **Manifests are also committed to git** in `data/manifests/<version>.json`, so the repo history records every dataset version. `data/manifest.json` (gitignored) records which version is checked out locally.
4. **`sync.py` enforces the rules:**
   - `publish` refuses an existing version;
   - `pull` verifies checksums and refuses to overwrite local files that differ from the target version unless `--force` is given;
   - `push` is removed;
   - `status` reports whether local files still match their version.
5. **Results carry their dataset version.** Training scripts stamp `dataset_version` on prediction CSVs, and `db/load_parquet.py` tags model runs `clean_<version>` (`clean_<version>_trends`). Prediction files from before this ADR are v1.
6. **v1 = the S3 flat prefix exactly as it was on 2026-09-27,** copied server-side with its anomalies documented in the manifest's notes rather than fixed. The flat prefix is left in place but no longer written to.

## Rationale

- Immutability is what makes "the model was trained on v2" a checkable statement, and it makes it impossible to lose a dataset by publishing a newer one.
- The manifest captures what a date alone can't: the code, the rules and the exact files.
- Tracking manifests in git ties data versions to code history without adding a tool.

## Alternatives considered

- **Date-named versions (e.g. `2026-09-27`)** — dates don't say what changed, and two builds on one day would collide. The fetch date is in the manifest instead.
- **DVC** — links each commit to data versions automatically, but adds a tool and a workflow for a one-person project; can be revisited if collaboration grows.
- **Delta Lake / Iceberg / lakeFS** — time travel and branching over tables; overkill for a handful of parquet files.
- **S3 bucket versioning alone** — protects against accidental overwrites but gives no names, descriptions or checksums. Enabled on 2026-09-27 as an extra safety net (it can be suspended but not removed; old object versions add a little storage cost); not a substitute.

## Assumptions

1. Versions are small enough to keep in full (v1 ≈ 490 MB). If storage cost matters later, drop intermediate files from old versions but keep the clean files and manifests.
2. Publishing is the only way data reaches S3. Anything written to S3 another way is outside this scheme.

## Consequences

- **Workflow:** after a pipeline run, `python data/sync.py publish v<N+1> --parent v<N> --notes "…"`, then commit `data/manifests/v<N+1>.json`, then train.
- Training on unpublished data works but stamps `dataset_version = "unversioned"` and prints a warning.
- `/sync-and-train` and the README were updated; ADR-012's storage choice (S3 + local parquet) stands, only its layout is replaced.

## Related ADRs

- ADR-012: S3 + DuckDB data lake (layout superseded here)
- ADR-013: Keyset pagination + full re-fetch (why v2 is a rebuild, not an append)
- ADR-015: Evaluation methodology (results must name the dataset version they were scored on)
