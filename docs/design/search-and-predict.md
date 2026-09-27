# Design: Search-and-Predict + Live Track Record

**Status:** Draft, 2026-09-27 · **Decision record:** ADR-017 · **Blocked on:** ADR-018 (volume feature), v2 retrain

## Goal

A public portfolio site where anyone can look up an open Polymarket market and see what each of our models predicts **right now** next to the market price. Every prediction is stored, and once the market resolves, it is scored. Over time the site builds an honest **live track record**.

## Screens (first version)

1. **Search / predict** — the main page.
   - One input that accepts a market URL, market slug, event URL, condition ID (`0x…`) or free-text name.
   - An event with several markets (e.g. 33 candidates for Ethiopian PM) shows a picker.
   - Result:
     - the market price and each model's probability and gap (model − market);
     - one headline model (the best on the latest dataset version, per `rescore_paper.py`), with the other five listed below;
     - a price-history chart with the models' "now" point;
     - any warnings;
     - links to Polymarket and to each model's track record.
2. **Model track record** — backtest results on the latest dataset version (ADR-015 method), and the **live** results from stored lookups once enough markets have resolved.
3. **Research** — the project story: research questions, method, results.

(The earlier "live market list" of all ~6,700 scoreable open markets was dropped: a search box is simpler to build and run, and just as useful for a portfolio piece.)

## Input resolution (Gamma API — verified 2026-09-27)

| Input | Lookup |
|---|---|
| `polymarket.com/event/<event>/<market>` or a market slug | `GET /markets?slug=<market>` |
| `polymarket.com/event/<event>` | `GET /events?slug=<event>`, then pick a market |
| `0x…` condition ID | `GET /markets?condition_ids=<id>&closed=…` |
| Free text | `GET /public-search?q=…`, then pick a result |

## Eligibility (ADR-017)

| Condition | Why it matters | Behaviour |
|---|---|---|
| Not binary | Models only predict YES/NO | **Reject** with explanation |
| Already closed | Nothing to predict | Reject; link to the outcome |
| Age < 14 days | Rolling features lack history (ADR-002 burn-in) | Predict + warning |
| < 14 days to scheduled end | Outside training window (ADR-002 cutoff) | Predict + warning |
| Price ≥ 0.95 or ≤ 0.05 | Such rows were removed from training (ADR-014) | Predict + warning |
| Scheduled duration < 30 days, volume < $1k | Outside market filters (ADR-008) | Predict + warning |
| No current Google Trends week for the category | Trends features stale | Predict + warning on the Trends models |

Warnings are stored with the prediction so the track record can be reported with and without warned predictions.

## Prediction path

1. Resolve the input to one market: Gamma metadata and the CLOB token ID.
2. Fetch the full CLOB price history (same call as `build_snapshots.fetch_price_history`).
3. **Compute features as of now, with the training code.** Refactor the per-snapshot feature block of `build_snapshots.compute_snapshots` into a shared `features_at(market, price_df, ts)` used by both the pipeline and the API. Features computed separately at training and serving time would drift apart silently.
4. Look up the category (cached per market; one Claude Haiku call via the `categorize_markets` prompt for unseen markets) and the latest Trends week.
5. Run all six models (LR, XGBoost, RF, each with and without Trends) from the artifacts of the current production dataset version.
6. Store the lookup (below) and return the result. Repeat lookups of the same market within a few hours reuse the stored snapshot.

Target latency is ~1–2 s: two Polymarket calls, an optional LLM call and in-memory inference.

## API contract (FastAPI)

```
GET /resolve?q=<url|slug|id|text>
  → { "markets": [ { "market_id", "question", "slug", "end_date", "price", "binary" }, … ] }

GET /predict?market_id=<condition id>
  → {
      "market":   { "market_id", "question", "category", "url", "start_date", "end_date",
                    "price", "volume_to_date" },
      "as_of":    "<ISO timestamp>",
      "dataset_version": "v2",
      "headline_model": "rf_trends",
      "predictions": [ { "model": "rf_trends", "probability": 0.34, "gap": +0.09,
                         "warnings": [] }, … six entries … ],
      "warnings": [ { "code": "near_close", "message": "…" } ],
      "price_history": [ { "t": "<ISO>", "p": 0.21 }, … ]
    }
  400 { "error": "not_binary" | "closed" | "not_found" }

GET /track-record
  → backtest metrics per model (from model_runs) + live metrics over resolved lookups
```

## Storage — live track record (new migration)

The existing `snapshots` / `predictions` tables (ADR-010/011) are the base. Changes:

- `markets`: add `closed_time TIMESTAMPTZ`, `resolution_status VARCHAR` (`open` / `resolved` / `ambiguous` / `voided`), `resolved_at TIMESTAMPTZ`. `outcome` stays NULL until resolved.
- `snapshots`: add
  - `features JSONB` — the **complete** feature vector used, so any prediction can be reproduced;
  - `warnings JSONB`;
  - `source VARCHAR` (`search`);
  - `volume_to_date FLOAT`.
- `predictions`: unchanged. Each row points at a `model_run` (model × dataset version) and a snapshot.

## Resolution job

A scheduled job (daily; GitHub Actions cron is enough) that:

1. selects markets with `resolution_status = 'open'`;
2. looks them up by condition ID in batches of 100 (as `fix_leakage.fetch_closed_times` does);
3. for closed ones, sets `outcome` with the ADR-004 rule (final YES price ≥ 0.95 → YES, ≤ 0.05 → NO, otherwise `ambiguous`), plus `closed_time` and `resolved_at`.

## Live track record — how it will be scored (ADR-015 rules)

- **One prediction per market**, the first lookup, so popular markets don't dominate.
- All models scored on the same markets; AUC / log-loss / Brier vs the market price at lookup time, with market-level bootstrap CIs.
- Reported overall and without warned predictions.
- **Caveat shown on the page:** these are markets people chose to look up, not a random sample.
- Until ~100+ markets have resolved, show counts and individual outcomes rather than headline metrics.

## Before building

1. **ADR-018:** fix the volume feature, which currently uses each market's final lifetime volume. It's unavailable for live markets and leaks the future. Retrain after.
2. Retrain on v2 and pick the headline model from `rescore_paper.py`.
3. Refactor the shared feature function (step 3 above) and test that it reproduces training snapshots exactly.

## Open questions

- Hosting: the API needs the six model artifacts in memory (the RF forests may be large) — Render / Fly / Railway vs a small VM.
- Frontend stack (Next.js on Vercel is the current lean).
- Rate limiting and abuse protection for a public endpoint that calls Polymarket and an LLM.
