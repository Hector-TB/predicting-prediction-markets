-- Migration 001: initial schema
-- Five tables: markets, trends, model_runs, snapshots, predictions
-- snapshots + predictions are for live markets only (historical data stays in parquet — ADR-011)

-- ── markets ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS markets (
    market_id       VARCHAR PRIMARY KEY,
    clob_token_id   VARCHAR,
    question        TEXT,
    category        VARCHAR,
    start_date      TIMESTAMPTZ,
    end_date        TIMESTAMPTZ,
    duration_days   FLOAT,
    total_volume    FLOAT,
    yes_final_price FLOAT,
    outcome         SMALLINT,       -- 1=YES, 0=NO, NULL=unresolved
    split           VARCHAR(10),    -- 'train', 'test', NULL=live
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_split    ON markets(split);

-- ── trends ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trends (
    id              SERIAL PRIMARY KEY,
    category        VARCHAR NOT NULL,
    week_start      DATE    NOT NULL,
    trend_value     FLOAT,
    trend_ma4       FLOAT,
    trend_change_4w FLOAT,
    trend_spike     FLOAT,
    UNIQUE (category, week_start)
);

CREATE INDEX IF NOT EXISTS idx_trends_category ON trends(category);

-- ── model_runs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_runs (
    id          SERIAL PRIMARY KEY,
    model_name  VARCHAR NOT NULL,
    dataset_tag VARCHAR NOT NULL,
    notes       TEXT,
    metrics     JSONB,
    hyperparams JSONB,
    git_sha     VARCHAR(40),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_name, dataset_tag)
);

-- ── snapshots (live markets only) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS snapshots (
    id                   BIGSERIAL PRIMARY KEY,
    market_id            VARCHAR     NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    snapshot_timestamp   TIMESTAMPTZ NOT NULL,
    price_at_snapshot    FLOAT,
    log_volume           FLOAT,
    pct_lifetime_elapsed FLOAT,
    days_before_close    FLOAT,
    price_mean_7d        FLOAT,
    price_volatility_7d  FLOAT,
    price_change_7d      FLOAT,
    price_trend_7d       FLOAT,
    price_mean_14d       FLOAT,
    price_volatility_14d FLOAT,
    price_change_14d     FLOAT,
    price_trend_14d      FLOAT,
    UNIQUE (market_id, snapshot_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_ts
    ON snapshots(market_id, snapshot_timestamp DESC);

-- ── predictions (live markets only) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL PRIMARY KEY,
    snapshot_id  BIGINT  NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    model_run_id INTEGER NOT NULL REFERENCES model_runs(id) ON DELETE CASCADE,
    probability  FLOAT   NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (snapshot_id, model_run_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_run ON predictions(model_run_id);
