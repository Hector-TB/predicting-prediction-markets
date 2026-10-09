-- Migration 006: store every /predict lookup (ADR-017, ADR-025)
-- A lookup = one `snapshots` row (features as of the lookup, warnings, price) plus one
-- `predictions` row per model of the release served. Markets looked up are upserted into
-- `markets` with split NULL; the daily resolution job fills outcome / closed_time later.
-- Needs migration 005 (model_runs.release): predictions point at the release's model_runs.

-- ── markets: resolution status for looked-up markets ─────────────────────────
ALTER TABLE markets
    ADD COLUMN IF NOT EXISTS slug              VARCHAR,
    ADD COLUMN IF NOT EXISTS closed_time       TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS resolution_status VARCHAR(10),   -- open | resolved | ambiguous | voided; NULL = dataset market
    ADD COLUMN IF NOT EXISTS resolved_at       TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_markets_resolution_status
    ON markets(resolution_status) WHERE resolution_status IS NOT NULL;

-- ── snapshots: the complete lookup ───────────────────────────────────────────
ALTER TABLE snapshots
    ADD COLUMN IF NOT EXISTS features       JSONB,          -- every model input, as fed to the models (ADR-025 fills included)
    ADD COLUMN IF NOT EXISTS warnings       JSONB,          -- ADR-025 codes, e.g. ["near_close"]
    ADD COLUMN IF NOT EXISTS market_price   FLOAT,          -- Gamma's YES price at the lookup (the gap is measured against it)
    ADD COLUMN IF NOT EXISTS end_date_seen  VARCHAR,        -- Gamma's endDate exactly as seen at the lookup (NULL if absent)
    ADD COLUMN IF NOT EXISTS volume_to_date FLOAT,
    ADD COLUMN IF NOT EXISTS release        VARCHAR,        -- model release served (ADR-019)
    ADD COLUMN IF NOT EXISTS source         VARCHAR(20) DEFAULT 'search';

-- ── row-level security ──────────────────────────────────────────────────────
-- The API writes with the database owner (DATABASE_URL), which bypasses RLS. With RLS on
-- and no policies, Supabase's public REST key can neither read nor write these tables.
ALTER TABLE markets     ENABLE ROW LEVEL SECURITY;
ALTER TABLE trends      ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_runs  ENABLE ROW LEVEL SECURITY;
ALTER TABLE snapshots   ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
