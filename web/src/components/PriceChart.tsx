import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { date, pct } from '../format'

interface Props {
  history: { t: string; p: number }[]
  asOf: string
  marketPrice: number | null
  model: { name: string; probability: number }
}

const H = 260
const M = { top: 12, right: 112, bottom: 28, left: 44 }
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1]
const DAY = 86_400_000

/** Clean time ticks: month starts for long spans, round day steps for short ones. */
function timeTicks(t0: number, t1: number, maxTicks: number): { t: number; label: string }[] {
  const span = t1 - t0
  const out: { t: number; label: string }[] = []
  if (span > 75 * DAY) {
    const months = span / (30 * DAY)
    const step = [1, 2, 3, 6, 12].find((s) => months / s <= maxTicks) ?? 12
    const d = new Date(t0)
    d.setUTCDate(1)
    d.setUTCHours(0, 0, 0, 0)
    d.setUTCMonth(d.getUTCMonth() + 1)
    while (d.getUTCMonth() % step !== 0) d.setUTCMonth(d.getUTCMonth() + 1)
    for (; d.getTime() <= t1; d.setUTCMonth(d.getUTCMonth() + step)) {
      const jan = d.getUTCMonth() === 0
      out.push({
        t: d.getTime(),
        label: d.toLocaleDateString(undefined, jan || step >= 6 ? { month: 'short', year: 'numeric' } : { month: 'short', timeZone: 'UTC' }),
      })
    }
  } else {
    const days = span / DAY
    const step = [1, 2, 7, 14, 28].find((s) => days / s <= maxTicks) ?? 28
    const d = new Date(t0)
    d.setUTCHours(0, 0, 0, 0)
    d.setUTCDate(d.getUTCDate() + 1)
    for (; d.getTime() <= t1; d.setUTCDate(d.getUTCDate() + step)) {
      out.push({ t: d.getTime(), label: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' }) })
    }
  }
  return out
}

/** Place two end labels; if they'd collide, spread them apart (the caller draws leader lines). */
function placeLabels(a: number, b: number, gap = 18): [number, number] {
  if (Math.abs(a - b) >= gap) return [a, b]
  const mid = (a + b) / 2
  return a <= b ? [mid - gap / 2, mid + gap / 2] : [mid + gap / 2, mid - gap / 2]
}

export default function PriceChart({ history, asOf, marketPrice, model }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(720)
  const [hover, setHover] = useState<number | null>(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const pts = useMemo(() => history.map((h) => ({ t: Date.parse(h.t), p: h.p })), [history])
  const tNow = Date.parse(asOf)
  const t0 = pts.length ? pts[0].t : tNow - DAY
  const iw = width - M.left - M.right
  const ih = H - M.top - M.bottom
  const x = (t: number) => M.left + ((t - t0) / Math.max(tNow - t0, 1)) * iw
  const y = (p: number) => M.top + (1 - p) * ih

  const line = pts.map((d, i) => `${i ? 'L' : 'M'}${x(d.t).toFixed(1)},${y(d.p).toFixed(1)}`).join('')
  const area = pts.length ? `${line}L${x(pts[pts.length - 1].t).toFixed(1)},${y(0)}L${x(pts[0].t).toFixed(1)},${y(0)}Z` : ''
  const ticks = timeTicks(t0, tNow, Math.max(2, Math.floor(iw / 90)))

  const last = pts[pts.length - 1]
  const endPrice = marketPrice ?? last?.p ?? null
  const yMarket = endPrice != null ? y(endPrice) : null
  const yModel = y(model.probability)
  const [lMarket, lModel] = yMarket != null ? placeLabels(yMarket, yModel) : [null, yModel]
  const xEnd = x(tNow)

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!pts.length) return
    const box = e.currentTarget.getBoundingClientRect()
    const t = t0 + ((e.clientX - box.left - M.left) / iw) * (tNow - t0)
    let lo = 0
    let hi = pts.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (pts[mid].t < t) lo = mid + 1
      else hi = mid
    }
    const i = lo > 0 && t - pts[lo - 1].t < pts[lo].t - t ? lo - 1 : lo
    setHover(i)
  }
  const h = hover != null ? pts[hover] : null

  return (
    <div className="chart" ref={ref}>
      <div className="legend">
        <span>
          <i className="key line" style={{ background: 'var(--series-1)' }} /> Market price of YES
        </span>
        <span>
          <i className="key" style={{ background: 'var(--series-2)' }} /> {model.name} now
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${H}`}
        width={width}
        height={H}
        role="img"
        aria-label={`Market price of YES over time, ${pct(endPrice)} now; ${model.name} estimates ${pct(model.probability)}.`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {Y_TICKS.map((v) => (
          <g key={v}>
            <line x1={M.left} x2={M.left + iw} y1={y(v)} y2={y(v)} stroke={v === 0 ? 'var(--axis)' : 'var(--grid)'} strokeWidth={1} />
            <text x={M.left - 8} y={y(v)} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--ink-muted)" style={{ fontVariantNumeric: 'tabular-nums' }}>
              {v * 100}%
            </text>
          </g>
        ))}
        {ticks.map((tk) => (
          <text key={tk.t} x={x(tk.t)} y={H - 8} textAnchor="middle" fontSize={12} fill="var(--ink-muted)">
            {tk.label}
          </text>
        ))}

        <path d={area} fill="var(--series-1)" opacity={0.1} />
        <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

        {/* now: market price and the model's estimate */}
        {yMarket != null && lMarket != null && (
          <>
            {last && <line x1={x(last.t)} x2={xEnd} y1={y(last.p)} y2={yMarket} stroke="var(--series-1)" strokeWidth={2} opacity={0.5} />}
            <circle cx={xEnd} cy={yMarket} r={4} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
            {lMarket !== yMarket && <line x1={xEnd + 6} x2={xEnd + 14} y1={yMarket} y2={lMarket} stroke="var(--axis)" />}
            <text x={xEnd + 16} y={lMarket} dy="0.32em" fontSize={12} fill="var(--ink-2)">
              Market <tspan fontWeight={600} fill="var(--ink)">{pct(endPrice)}</tspan>
            </text>
          </>
        )}
        <circle cx={xEnd} cy={yModel} r={5} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
        {lModel !== yModel && <line x1={xEnd + 6} x2={xEnd + 14} y1={yModel} y2={lModel} stroke="var(--axis)" />}
        <text x={xEnd + 16} y={lModel} dy="0.32em" fontSize={12} fill="var(--ink-2)">
          {model.name} <tspan fontWeight={600} fill="var(--ink)">{pct(model.probability)}</tspan>
        </text>

        {h && (
          <g pointerEvents="none">
            <line x1={x(h.t)} x2={x(h.t)} y1={M.top} y2={M.top + ih} stroke="var(--axis)" />
            <circle cx={x(h.t)} cy={y(h.p)} r={4} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        )}
        {/* generous hit area for hover */}
        <rect x={M.left} y={M.top} width={iw} height={ih} fill="transparent" />
      </svg>
      {h && (
        <div className="tooltip" style={{ left: x(h.t), top: y(h.p) }}>
          {date(new Date(h.t).toISOString())} · <b>{pct(h.p)}</b>
        </div>
      )}
      <details>
        <summary className="small">Show the price history as a table</summary>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th className="num">Price of YES</th>
            </tr>
          </thead>
          <tbody>
            {[...pts].reverse().map((d) => (
              <tr key={d.t}>
                <td>{new Date(d.t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</td>
                <td className="num">{pct(d.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}
