-- Migration 005: model_runs.release (ADR-019)
-- Rows written by `scripts/release.py promote` carry the release (r1, r2, …) whose
-- metrics they hold; rows from db/load_parquet.py (local prediction CSVs) have NULL.
-- Two releases can share a dataset version (e.g. a re-tune), so the release joins
-- the unique key. NULLS NOT DISTINCT keeps load_parquet's rows one per (model, dataset).

ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS release VARCHAR;

ALTER TABLE model_runs DROP CONSTRAINT IF EXISTS model_runs_model_name_dataset_tag_key;
ALTER TABLE model_runs
  ADD CONSTRAINT model_runs_model_name_dataset_tag_release_key
  UNIQUE NULLS NOT DISTINCT (model_name, dataset_tag, release);
