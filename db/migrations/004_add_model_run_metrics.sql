-- Migration 004: add metrics and hyperparams JSONB columns to model_runs
-- Apply via Supabase MCP (mcp__supabase__apply_migration) once the DB is healthy.

ALTER TABLE model_runs
  ADD COLUMN IF NOT EXISTS metrics    JSONB,
  ADD COLUMN IF NOT EXISTS hyperparams JSONB;
