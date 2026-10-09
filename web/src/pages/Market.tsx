import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, predict, type PredictResult } from '../api'
import PriceChart from '../components/PriceChart'
import { CATEGORY_NAMES, date, dateTime, modelName, money, pct, pts } from '../format'

type State =
  | { kind: 'loading' }
  | { kind: 'done'; r: PredictResult }
  | { kind: 'error'; code: string; message: string }

const ERROR_TEXT: Record<string, string> = {
  not_binary: 'Only YES/NO markets can be predicted, and this one has more outcomes.',
  closed: 'This market has closed, so there is nothing left to predict.',
  not_found: 'No Polymarket market has this ID.',
  upstream: 'Polymarket could not be reached. Try again in a moment.',
  offline: 'The prediction service could not be reached.',
}

export default function Market() {
  const { marketId = '' } = useParams()
  const [state, setState] = useState<State>({ kind: 'loading' })

  const run = useCallback(
    (fresh: boolean) => {
      setState({ kind: 'loading' })
      predict(marketId, fresh)
        .then((r) => setState({ kind: 'done', r }))
        .catch((e: ApiError) => setState({ kind: 'error', code: e.code, message: ERROR_TEXT[e.code] ?? e.message }))
    },
    [marketId],
  )

  useEffect(() => run(false), [run])

  useEffect(() => {
    document.title = state.kind === 'done' ? `${state.r.market.question} · Predicting Prediction Markets` : 'Predicting Prediction Markets'
  }, [state])

  if (state.kind === 'loading') {
    return (
      <div className="loading" aria-live="polite">
        <div className="spinner" />
        Fetching the price history and running the models…
      </div>
    )
  }
  if (state.kind === 'error') {
    return (
      <div className="card error section">
        <p>{state.message}</p>
        <p>
          <Link to="/">Search for another market</Link>
        </p>
      </div>
    )
  }
  return <Prediction r={state.r} onRefresh={() => run(true)} />
}

function Prediction({ r, onRefresh }: { r: PredictResult; onRefresh: () => void }) {
  const { market } = r
  const headline = r.predictions.find((p) => p.model === r.headline_model) ?? r.predictions[0]
  const name = modelName(headline.model)

  return (
    <>
      <section className="market-head">
        <div className="small muted">{CATEGORY_NAMES[market.category ?? ''] ?? 'Uncategorised'}</div>
        <h1>{market.question}</h1>
        <div className="meta">
          <span>{market.end_date ? `Ends ${date(market.end_date)}` : 'No end date'}</span>
          <span>{money(market.volume_to_date)} traded</span>
          {market.url && (
            <a href={market.url} target="_blank" rel="noreferrer">
              View on Polymarket ↗
            </a>
          )}
        </div>
      </section>

      <section className="tiles" aria-label="Prediction">
        <div className="tile">
          <div className="label">
            <i className="key line" style={{ background: 'var(--series-1)' }} /> Market price of YES
          </div>
          <div className="value">{pct(market.price)}</div>
          <div className="note">What traders are paying now</div>
        </div>
        <div className="tile">
          <div className="label">
            <i className="key" style={{ background: 'var(--series-2)' }} /> {name}
          </div>
          <div className="value">{pct(headline.probability)}</div>
          <div className="note">Chance of YES; our most accurate model in testing</div>
        </div>
        <div className="tile">
          <div className="label">Difference</div>
          <div className="value">{pts(headline.gap)}</div>
          <div className="note">
            {headline.gap == null ? 'No market price' : headline.gap >= 0 ? `${name} is above the market` : `${name} is below the market`}
          </div>
        </div>
      </section>

      {r.warnings.length > 0 && (
        <section className="warnings" aria-label="Caveats">
          {r.warnings.map((w) => (
            <div className="warning" key={w.code}>
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 3 2 21h20L12 3Z" fill="var(--warning)" />
                <path d="M12 10v5M12 17.5v.5" stroke="#0b0b0b" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <span>
                <strong>Caveat: </strong>
                {w.message}
              </span>
            </div>
          ))}
        </section>
      )}

      <section className="card section">
        <h2>Price history</h2>
        {r.price_history.length ? (
          <PriceChart history={r.price_history} asOf={r.as_of} marketPrice={market.price} model={{ name, probability: headline.probability }} />
        ) : (
          <p className="secondary">This market has no trading history yet.</p>
        )}
      </section>

      <section className="card section">
        <h2>Every model</h2>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Chance of YES</th>
              <th className="num">vs market</th>
            </tr>
          </thead>
          <tbody>
            {r.predictions.map((p) => (
              <tr key={p.model} className={p.model === r.headline_model ? 'headline' : undefined}>
                <td>{modelName(p.model)}</td>
                <td className="num">{pct(p.probability)}</td>
                <td className="num">{pts(p.gap)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted">
          In backtests on markets that resolved after training, the tree models were slightly more
          accurate than the market price on average, mostly 30 or more days before close. Any single
          prediction can be far off.
        </p>
        <details>
          <summary>What the models saw</summary>
          <table>
            <tbody>
              {Object.entries(r.features).map(([k, v]) => (
                <tr key={k}>
                  <td>{k}</td>
                  <td className="num">{typeof v === 'number' ? +v.toFixed(4) : (v ?? '—')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </section>

      <section className="section small secondary">
        <p>
          Predicted {dateTime(r.as_of)} with model release {r.release} (trained on dataset {r.dataset_version}).{' '}
          {r.stored
            ? `Saved as lookup #${r.lookup_id}: it will be scored against the real outcome when the market resolves.`
            : 'This prediction could not be saved, so it will not count toward the track record.'}
        </p>
        <div>
          <button className="button ghost" onClick={onRefresh}>
            Predict again
          </button>
        </div>
      </section>
    </>
  )
}
