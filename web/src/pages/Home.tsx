import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ApiError, resolve, type MarketSummary } from '../api'
import { date, pct } from '../format'

const EXAMPLES = ['Fed rate cut', 'Bitcoin', 'https://polymarket.com/event/how-many-fed-rate-cuts-in-2026']

type State =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; markets: MarketSummary[] }
  | { kind: 'error'; message: string }

export default function Home() {
  const [params, setParams] = useSearchParams()
  const q = params.get('q') ?? ''
  const [input, setInput] = useState(q)
  const [state, setState] = useState<State>({ kind: 'idle' })

  useEffect(() => {
    setInput(q)
    if (!q) {
      setState({ kind: 'idle' })
      return
    }
    const ctrl = new AbortController()
    setState({ kind: 'loading' })
    resolve(q, ctrl.signal)
      .then((r) => setState({ kind: 'done', markets: r.markets }))
      .catch((e) => {
        if (e.name !== 'AbortError') setState({ kind: 'error', message: (e as ApiError).message })
      })
    return () => ctrl.abort()
  }, [q])

  const search = (text: string) => {
    const t = text.trim()
    if (t) setParams({ q: t })
  }

  return (
    <>
      <section className="hero">
        <h1>What do the models think?</h1>
        <p>
          Search any open Polymarket market. Three models trained on tens of thousands of resolved
          markets estimate the chance of YES, next to the market's own price.
        </p>
        <form
          className="search"
          role="search"
          onSubmit={(e) => {
            e.preventDefault()
            search(input)
          }}
        >
          <input
            type="search"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Paste a Polymarket link, or search by name"
            aria-label="Market link or name"
            autoFocus
          />
          <button className="button" type="submit" disabled={!input.trim()}>
            Search
          </button>
        </form>
        {!q && (
          <div className="examples">
            <span className="muted">Try</span>
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip" onClick={() => search(ex)}>
                {ex.startsWith('http') ? 'a Polymarket link' : ex}
              </button>
            ))}
          </div>
        )}
      </section>

      {state.kind === 'loading' && (
        <div className="loading" aria-live="polite">
          <div className="spinner" />
          Searching Polymarket…
        </div>
      )}
      {state.kind === 'error' && <div className="card error">{state.message}</div>}
      {state.kind === 'done' && <Results markets={state.markets} />}
    </>
  )
}

function Results({ markets }: { markets: MarketSummary[] }) {
  if (markets.length === 0) {
    return <p className="secondary">No open markets found. Try a different name, or paste the market's link.</p>
  }
  // Predictable markets first: open and binary
  const sorted = [...markets].sort((a, b) => Number(canPredict(b)) - Number(canPredict(a)))
  return (
    <section className="section" aria-label="Results">
      <h2>
        {markets.length} market{markets.length === 1 ? '' : 's'}
      </h2>
      <ul className="results">
        {sorted.map((m) => (
          <li key={m.market_id}>
            <ResultRow m={m} />
          </li>
        ))}
      </ul>
    </section>
  )
}

const canPredict = (m: MarketSummary) => m.binary && !m.closed

function ResultRow({ m }: { m: MarketSummary }) {
  const body = (
    <>
      <div>
        <div className="q">
          {m.question}
          {m.closed && <span className="badge">Closed</span>}
          {!m.binary && <span className="badge">Not YES/NO</span>}
        </div>
        <div className="meta">{m.end_date ? `Ends ${date(m.end_date)}` : 'No end date'}</div>
      </div>
      <div className="price">
        {pct(m.price, 0)}
        <small>market</small>
      </div>
    </>
  )
  if (!canPredict(m)) {
    return (
      <div className="result disabled" title="Only open YES/NO markets can be predicted">
        {body}
      </div>
    )
  }
  return (
    <Link className="result" to={`/market/${m.market_id}`}>
      {body}
    </Link>
  )
}
