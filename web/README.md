# Web frontend

React + TypeScript (Vite). Search a Polymarket market, then see every production model's prediction next to the market price, with ADR-025's caveats.

```bash
# terminal 1, repo root: the API
python3 -m uvicorn api.main:app --reload --port 8000

# terminal 2: the site, http://localhost:5173 (/api is proxied to port 8000)
cd web && npm install && npm run dev
```

Every market page view calls `/predict`, which **stores a lookup in Supabase** (the live track record). To try the site without storing anything, start the API with `DATABASE_URL= python3 -m uvicorn api.main:app`; pages then say the prediction wasn't saved. The client reuses a market's prediction for 10 minutes (`src/api.ts`), so re-renders and back-navigation don't store duplicates; "Predict again" makes a new lookup.

`npm run build` type-checks and builds into `dist/`; `npm run lint` runs oxlint.
