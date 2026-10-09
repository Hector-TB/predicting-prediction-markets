export const MODEL_NAMES: Record<string, string> = {
  gradient_boosting: 'XGBoost',
  random_forest: 'Random forest',
  logistic_regression: 'Logistic regression',
}

export const modelName = (m: string) => MODEL_NAMES[m] ?? m

/** A probability as a percentage: 0.1662 → "16.6%". */
export const pct = (p: number | null | undefined, digits = 1) =>
  p == null ? '—' : `${(p * 100).toFixed(digits)}%`

/** A gap in percentage points, always signed: 0.0112 → "+1.1 pts". */
export const pts = (d: number | null | undefined) =>
  d == null ? '—' : `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d * 100).toFixed(1)} pts`

export const date = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : '—'

export const dateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })

export const money = (v: number) =>
  new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD', notation: 'compact', maximumFractionDigits: 1 }).format(v)

export const CATEGORY_NAMES: Record<string, string> = {
  politics_us: 'US politics', politics_global: 'Global politics', crypto: 'Crypto', sports: 'Sports',
  finance: 'Finance', geopolitics: 'Geopolitics', science_tech: 'Science & tech',
  entertainment: 'Entertainment', other: 'Other',
}
