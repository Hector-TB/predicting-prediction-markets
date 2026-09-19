# ADR-012: S3 + DuckDB as the Data Lake

**Date:** 2026-09-19  
**Status:** Accepted  
**Deciders:** Hector Thompson Baroni  

---

## Context

The `data/` directory contains ~380 MB of parquet files tracked in git. This causes slow clones, GitHub LFS pressure, and makes it impossible for a second developer (or a CI machine) to work without the full binary history. The files are also static artefacts derived from the pipeline — they should be stored in object storage, not a code repository.

## Decision

**S3 is the source of truth for all data files. Parquet files are removed from git.**

- All `data/*.parquet` and `data/polymarket_markets_meta.csv` files are uploaded to a private S3 bucket and removed from git tracking.
- `data/sync.py` is the single interface for push/pull operations (boto3-based, reads credentials from `.env`).
- **DuckDB** is added as a dependency for ad-hoc analytical queries against parquet files — locally or directly against S3 via the `httpfs` extension.
- Training scripts are **unchanged** — they still read local parquet via pandas. `sync.py pull` fetches the data before training.

## Rationale

**S3 is the standard.** Object storage is purpose-built for large binary files: versioning, lifecycle policies, cross-region replication, presigned URLs. Cost is negligible (~$0.023/GB/month; the dataset is ~380 MB ≈ $0.01/month).

**DuckDB for analytics.** DuckDB can query parquet files directly with SQL — both local files and S3 paths — with no server required. It's faster than pandas for aggregations and joins, integrates naturally with pandas (`conn.execute(...).df()`), and can serve as a local data warehouse layer as the dataset grows. It also makes it trivial to query S3 directly from a notebook without downloading the full file.

**Training scripts unchanged.** Replacing pandas `read_parquet` calls with network reads would add latency and a network dependency to every training run. The `sync.py` pattern separates data acquisition (once) from training (fast, local).

## S3 bucket setup

1. AWS Console → S3 → Create bucket
   - Name: globally unique, e.g., `polymarket-ml-data-{initials}`  
   - Region: `us-east-1` (or closest)
   - Block all public access: **ON**
2. AWS Console → IAM → Users → Create user → Attach `AmazonS3FullAccess`
   - Create access key → "Application running outside AWS"
3. Add to `.env`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `AWS_REGION`

## Migration steps (one-time)

```bash
# 1. Upload existing files
python data/sync.py push

# 2. Verify everything uploaded
python data/sync.py status

# 3. Remove from git tracking (keep local copies)
git rm --cached data/*.parquet

# 4. Commit
git add .gitignore && git commit -m "Move data files to S3 (ADR-012)"
```

## DuckDB usage examples

```python
import duckdb

# Query local parquet
conn = duckdb.connect()
df = conn.execute("""
    SELECT category, COUNT(*) as n, AVG(outcome) as yes_rate
    FROM read_parquet('data/polymarket_ml_dataset_clean.parquet')
    GROUP BY category ORDER BY n DESC
""").df()

# Query S3 directly (no local download)
conn.execute("INSTALL httpfs; LOAD httpfs")
conn.execute("""
    SET s3_region = 'us-east-1';
    SET s3_access_key_id = '...';
    SET s3_secret_access_key = '...';
""")
df = conn.execute("SELECT * FROM read_parquet('s3://your-bucket/data/*.parquet') LIMIT 1000").df()
```

## Consequences

- `data/*.parquet` and `data/*.csv` are gitignored; git history is clean going forward.
- New machine setup: `pip install -e ".[dev]"` then `python data/sync.py pull`.
- The pipeline's output parquet files must be pushed to S3 after any pipeline re-run.
- `data/sync.py` uses size-based comparison (not ETag/hash) — sufficient for this workflow since files are only written by the pipeline, not edited in place.

## Related ADRs

- ADR-011: Offline/online split (parquet for training, Postgres for operations)
