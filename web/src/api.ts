// Client for the FastAPI backend (api/main.py); /api is proxied to it in development.

export interface MarketSummary {
  market_id: string
  question: string
  slug: string | null
  end_date: string | null
  price: number | null
  binary: boolean
  closed: boolean
}

export interface ModelPrediction {
  model: string
  probability: number
  gap: number | null
}

export interface Warning {
  code: string
  message: string
}

export interface PredictResult {
  market: {
    market_id: string
    question: string
    category: string | null
    url: string | null
    start_date: string
    end_date: string | null
    price: number | null
    volume_to_date: number
  }
  as_of: string
  release: string
  dataset_version: string
  headline_model: string
  predictions: ModelPrediction[]
  warnings: Warning[]
  features: Record<string, number | string | null>
  price_history: { t: string; p: number }[]
  lookup_id?: number
  stored?: boolean
}

export class ApiError extends Error {
  code: string
  constructor(code: string, message: string) {
    super(message)
    this.code = code
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, { signal })
  } catch (e) {
    if ((e as Error).name === 'AbortError') throw e
    throw new ApiError('offline', 'The prediction service could not be reached.')
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(body?.error ?? 'error', body?.message ?? `Request failed (${res.status}).`)
  }
  return body as T
}

export const resolve = (q: string, signal?: AbortSignal) =>
  get<{ markets: MarketSummary[] }>(`/resolve?q=${encodeURIComponent(q)}`, signal)

// Every /predict call is stored as a lookup (api/store.py), so one page view must make one
// call: StrictMode re-runs effects and back-navigation remounts pages. Reuse a pending or
// recent prediction of the same market instead; "Predict again" passes fresh = true.
const PREDICTION_TTL_MS = 10 * 60 * 1000
const predictions = new Map<string, { at: number; promise: Promise<PredictResult> }>()

export function predict(marketId: string, fresh = false): Promise<PredictResult> {
  const hit = predictions.get(marketId)
  if (hit && !fresh && Date.now() - hit.at < PREDICTION_TTL_MS) return hit.promise
  const promise = get<PredictResult>(`/predict?market_id=${encodeURIComponent(marketId)}`)
  promise.catch(() => predictions.delete(marketId))
  predictions.set(marketId, { at: Date.now(), promise })
  return promise
}
