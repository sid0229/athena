import { ChartColumn, Info } from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '../api'
import type { Analysis, Decision, Meta, Tier } from '../types'
import { disp } from '../lib'
import { Card, Empty, PageHeader, Spinner, cx } from '../components/ui'

// Colours come from the design tokens, so light and dark mode both follow the theme.
const C = {
  confirm: 'var(--ok)', override: 'var(--critical)', bar: 'var(--accent)',
  continued: 'var(--faint)', dose: 'var(--hold)', omitted: 'var(--review)', held: 'var(--muted)',
  absent: 'var(--line-strong)', new: 'var(--info)',
  interaction: 'var(--review)', none: 'var(--ok)', notCovered: 'var(--line-strong)',
}
const TIERS: Tier[] = ['critical', 'review', 'info']

/** Hover card shared by every chart on the page. */
type Tip = { x: number; y: number; body: ReactNode } | null
function useTip() {
  const [tip, setTip] = useState<Tip>(null)
  const on = (body: ReactNode) => ({
    onMouseMove: (e: React.MouseEvent) => setTip({ x: e.clientX, y: e.clientY, body }),
    onMouseLeave: () => setTip(null),
  })
  const el = tip && (
    <div role="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }}
      className="pointer-events-none fixed z-50 max-w-xs rounded-md border border-line-strong bg-surface px-3 py-2 text-sm text-ink shadow-lg">
      {tip.body}
    </div>
  )
  return { on, el }
}

export function Analytics({ a, meta, refreshKey }: { a: Analysis | null; meta: Meta; refreshKey: number }) {
  const [entries, setEntries] = useState<Decision[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.audit().then(r => { setEntries(r.entries); setErr(null) })
      .catch(e => setErr(`Could not load the audit log: ${(e as Error).message}`))
  }, [refreshKey])
  const tip = useTip()

  return (
    <div className="mx-auto max-w-[1320px]">
      <PageHeader title="Analytics" right={<span className="meta">All data stays on this device</span>} />
      {tip.el}

      <SectionTitle first title="Current case" sub={a ? 'From the note you analyzed last' : 'Analyze a note to see the case charts'} />
      {a ? <CaseCharts a={a} on={tip.on} /> : (
        <Empty icon={<ChartColumn size={24} strokeWidth={1.5} />} title="No case analyzed yet">
          Run a note from Med Input to see its admission → discharge flow and interaction coverage.
        </Empty>
      )}

      <SectionTitle title="Review activity" sub="Pharmacist decisions from the audit log (latest decision per finding)" />
      {err && <p role="alert" className="text-sm text-critical">{err}</p>}
      {!entries && !err && <Spinner />}
      {entries && <ReviewCharts entries={entries} meta={meta} on={tip.on} />}
    </div>
  )
}

const SectionTitle = ({ title, sub, first }: { title: string; sub: string; first?: boolean }) => (
  <div className={cx('mb-4 flex items-baseline gap-3', !first && 'mt-10')}>
    <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
    <span className="meta">{sub}</span>
  </div>
)

const ChartCard = ({ title, sub, children, className }: { title: string; sub?: ReactNode; children: ReactNode; className?: string }) => (
  <Card className={cx('px-5 py-4', className)}>
    <div className="mb-4"><h3 className="text-base font-semibold text-ink">{title}</h3>{sub && <p className="meta mt-0.5">{sub}</p>}</div>
    {children}
  </Card>
)

const Stat = ({ label, value, note }: { label: string; value: ReactNode; note?: string }) => (
  <Card className="px-5 py-4">
    <div className="meta">{label}</div>
    <div className="mt-1 text-3xl font-semibold tracking-tight text-ink">{value}</div>
    {note && <div className="mt-0.5 text-xs text-muted">{note}</div>}
  </Card>
)

const Legend = ({ items }: { items: [string, string][] }) => (
  <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1">
    {items.map(([label, color]) => (
      <span key={label} className="flex items-center gap-1.5 text-sm text-ink-2">
        <span className="size-2.5 rounded-sm" style={{ background: color }} aria-hidden />{label}
      </span>
    ))}
  </div>
)

type On = (body: ReactNode) => { onMouseMove: (e: React.MouseEvent) => void; onMouseLeave: () => void }

// ---------------------------------------------------------------- current case

type Flow = 'continued' | 'dose' | 'omitted' | 'held' | 'absent' | 'new'
const FLOW: Record<Flow, { label: string; color: string }> = {
  continued: { label: 'Continued', color: C.continued },
  dose: { label: 'Dose changed', color: C.dose },
  omitted: { label: 'Possible omission', color: C.omitted },
  held: { label: 'Stopped / held', color: C.held },
  absent: { label: 'Not listed (not reconciled)', color: C.absent },
  new: { label: 'New at discharge', color: C.new },
}

function CaseCharts({ a, on }: { a: Analysis; on: On }) {
  const c = a.report.meta.counts
  const flows = useMemo(() => {
    const ids = new Set(a.report.findings.map(f => f.id))
    const tier = Object.fromEntries(a.report.findings.map(f => [f.id, f.tier]))
    const out: Record<Flow, string[]> = { continued: [], dose: [], omitted: [], held: [], absent: [], new: [] }
    for (const m of a.report.medications) {
      const name = disp(m.name, a.display_names)
      if (m.in_admission && m.in_discharge) out[ids.has(`DOSE:${m.id}`) ? 'dose' : 'continued'].push(name)
      else if (m.in_admission) {   // same rules as Med Diff
        const t = tier[`OMIT:${m.id}`]
        out[!t ? 'absent' : t === 'info' ? 'held' : 'omitted'].push(name)
      } else if (m.in_discharge) out.new.push(name)
    }
    return out
  }, [a])
  const hasLists = a.report.meta.sections.some(s => s.kind === 'admission') && a.report.meta.sections.some(s => s.kind === 'discharge')
  const pairs = c.interactions + c.no_known_interaction + c.not_covered

  return (
    <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
      <ChartCard title="Admission → discharge" sub="How each admission medication was handled at discharge · hover a band for the drugs">
        {hasLists ? <FlowChart flows={flows} on={on} /> : (
          <div className="flex items-start gap-2 py-6 text-sm text-muted"><Info size={16} className="mt-0.5 shrink-0" />
            This note does not have both an admission and a discharge medication list, so there is no flow to show.</div>
        )}
      </ChartCard>
      <div className="flex flex-col gap-4">
        <ChartCard title="Findings by tier">
          <div className="grid grid-cols-3 gap-3">
            {TIERS.map(t => (
              <div key={t}>
                <div className={cx('text-2xl font-semibold', t === 'critical' ? 'text-critical' : t === 'review' ? 'text-review' : 'text-ink')}>{c.tiers[t] ?? 0}</div>
                <div className="meta capitalize">{t}</div>
              </div>
            ))}
          </div>
        </ChartCard>
        <ChartCard title="Interaction coverage" sub={`${pairs} drug pairs checked against DrugBank and DDInter`}>
          <Legend items={[[`Interaction (${c.interactions})`, C.interaction], [`No known interaction (${c.no_known_interaction})`, C.none], [`Not covered (${c.not_covered})`, C.notCovered]]} />
          <StackBar on={on} label="Drug pairs" parts={[
            ['Interaction', c.interactions, C.interaction], ['No known interaction', c.no_known_interaction, C.none],
            ['Not covered by either database', c.not_covered, C.notCovered]]} />
        </ChartCard>
      </div>
    </div>
  )
}

/** Two-column flow: admission list (left) to discharge list / not continued (right). */
function FlowChart({ flows, on }: { flows: Record<Flow, string[]>; on: On }) {
  const W = 720, H = 300, NODE = 14, GAP = 2, SPLIT = 18
  // Ordered so no bands cross: new drugs run along the top, dropped ones along the bottom.
  const leftOrder: Flow[] = ['continued', 'dose', 'omitted', 'held', 'absent']
  const rightOrder: Flow[] = ['new', 'continued', 'dose']
  const dropOrder: Flow[] = ['omitted', 'held', 'absent']
  const n = (f: Flow) => flows[f].length
  const adm = leftOrder.reduce((s, f) => s + n(f), 0)
  const dis = rightOrder.reduce((s, f) => s + n(f), 0)
  const dropped = dropOrder.reduce((s, f) => s + n(f), 0)
  const total = Math.max(adm + n('new'), dis + dropped, 1)
  const unit = (H - SPLIT - 40) / total
  // Left column: "not on admission" (new drugs) on top, the admission list below it.
  const ly: Partial<Record<Flow, number>> = { new: 20 }
  const admTop = 20 + n('new') * unit + (n('new') ? SPLIT : 0)
  let y = admTop
  for (const f of leftOrder) { ly[f] = y; y += n(f) * unit }
  // Right column: the discharge list on top, "not on discharge list" below it.
  const ry: Partial<Record<Flow, number>> = {}
  y = 20
  for (const f of rightOrder) { ry[f] = y; y += n(f) * unit }
  const dropTop = y + (dropped ? SPLIT : 0)
  y = dropTop
  for (const f of dropOrder) { ry[f] = y; y += n(f) * unit }

  const x0 = 150, x1 = W - 170
  const band = (f: Flow) => {
    const h = n(f) * unit
    if (!h) return null
    const yl = ly[f]!, yr = ry[f]!, mx = (x0 + x1) / 2
    const g = h > GAP * 2 ? GAP / 2 : 0
    const d = `M${x0 + NODE},${yl + g} C${mx},${yl + g} ${mx},${yr + g} ${x1},${yr + g} L${x1},${yr + h - g} C${mx},${yr + h - g} ${mx},${yl + h - g} ${x0 + NODE},${yl + h - g} Z`
    return (
      <path key={f} d={d} fill={FLOW[f].color} fillOpacity={0.55} className="cursor-default transition-[fill-opacity] hover:[fill-opacity:0.85]"
        {...on(<><div className="font-medium">{FLOW[f].label} · {n(f)}</div><div className="mt-1 text-muted">{flows[f].join(', ')}</div></>)} />
    )
  }
  const node = (x: number, y0: number, h: number, label: string, count: number, side: 'l' | 'r') => h > 0 && (
    <g>
      <rect x={x} y={y0} width={NODE} height={Math.max(h, 2)} rx={3} fill="var(--ink-2)" />
      <text x={side === 'l' ? x - 10 : x + NODE + 10} y={y0 + h / 2} dy="0.35em" textAnchor={side === 'l' ? 'end' : 'start'}
        className="fill-[var(--ink)] text-[13px] font-medium">{label}</text>
      <text x={side === 'l' ? x - 10 : x + NODE + 10} y={y0 + h / 2 + 16} dy="0.35em" textAnchor={side === 'l' ? 'end' : 'start'}
        className="fill-[var(--muted)] text-[12px]">{count} drug{count === 1 ? '' : 's'}</text>
    </g>
  )
  const summary = (Object.keys(FLOW) as Flow[]).filter(f => n(f)).map(f => `${FLOW[f].label}: ${n(f)}`).join('; ')

  return (
    <>
      <Legend items={(Object.keys(FLOW) as Flow[]).filter(f => n(f)).map(f => [`${FLOW[f].label} (${n(f)})`, FLOW[f].color])} />
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Admission to discharge flow. ${summary}`}>
        {(Object.keys(FLOW) as Flow[]).map(band)}
        {node(x0, 20, n('new') * unit, 'Not on admission', n('new'), 'l')}
        {node(x0, admTop, adm * unit, 'Admission list', adm, 'l')}
        {node(x1, 20, dis * unit, 'Discharge list', dis, 'r')}
        {node(x1, dropTop, dropped * unit, 'Not on discharge list', dropped, 'r')}
      </svg>
    </>
  )
}

function StackBar({ parts, label, on }: { parts: [string, number, string][]; label: string; on: On }) {
  const total = parts.reduce((s, p) => s + p[1], 0)
  if (!total) return <div className="meta">No data</div>
  return (
    <div role="img" aria-label={`${label}: ${parts.map(p => `${p[0]} ${p[1]}`).join(', ')}`}
      className="flex h-6 w-full gap-[2px] overflow-hidden rounded">
      {parts.filter(p => p[1] > 0).map(([name, v, color]) => (
        <div key={name} style={{ width: `${(v / total) * 100}%`, background: color }} className="h-full min-w-[3px]"
          {...on(<><span className="font-medium">{name}</span> · {v} ({Math.round((v / total) * 100)}%)</>)} />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- review activity

function ReviewCharts({ entries, meta, on }: { entries: Decision[]; meta: Meta; on: On }) {
  const latest = useMemo(() => {   // entries arrive newest first: keep the first decision per finding
    const seen = new Set<string>()
    return entries.filter(e => e.action === 'confirm' || e.action === 'override').filter(e => {
      const k = `${e.case_id}|${e.report_hash}|${e.finding_id}`
      if (seen.has(k)) return false
      seen.add(k); return true
    })
  }, [entries])
  if (!latest.length) return (
    <Empty icon={<ChartColumn size={24} strokeWidth={1.5} />} title="No decisions recorded yet">
      Confirm or override findings in Med Review or Med Diff and they will be summarised here.
    </Empty>
  )
  const confirmed = latest.filter(e => e.action === 'confirm').length
  const overridden = latest.length - confirmed
  const cases = new Set(latest.map(e => e.case_id)).size
  const reviewers = new Set(latest.map(e => e.reviewer)).size
  const nCorrections = entries.filter(e => e.action === 'correct').length

  const byTier = TIERS.map(t => {
    const rows = latest.filter(e => e.tier === t)
    return { t, confirm: rows.filter(e => e.action === 'confirm').length, override: rows.filter(e => e.action === 'override').length }
  })
  const tierMax = Math.max(1, ...byTier.map(r => r.confirm + r.override))

  const reasons = Object.entries(latest.filter(e => e.action === 'override').reduce<Record<string, number>>((m, e) => {
    const k = e.reason_code ?? 'other'; m[k] = (m[k] ?? 0) + 1; return m
  }, {})).sort((x, y) => y[1] - x[1])
  const reasonMax = Math.max(1, ...reasons.map(r => r[1]))

  const days = Array.from({ length: 14 }, (_, i) => {   // reviewer's local days, oldest first
    const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - (13 - i)); return d
  })
  const perDay = days.map(d => latest.filter(e => new Date(e.ts).toDateString() === d.toDateString()).length)
  const dayMax = Math.max(1, ...perDay)

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Findings decided" value={latest.length} note={`${entries.length} audit entries in total`} />
        <Stat label="Confirmed" value={`${Math.round((confirmed / latest.length) * 100)}%`} note={`${confirmed} of ${latest.length}`} />
        <Stat label="Overridden" value={overridden} note="each with a clinical reason" />
        <Stat label="Cases reviewed" value={cases} note={`${reviewers} reviewer${reviewers === 1 ? '' : 's'} · ${nCorrections} list correction${nCorrections === 1 ? '' : 's'}`} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Decisions by tier" sub="Does the pharmacist agree with Athena's priority?">
          <Legend items={[['Confirmed', C.confirm], ['Overridden', C.override]]} />
          <div className="flex flex-col gap-3">
            {byTier.map(r => (
              <div key={r.t} className="grid grid-cols-[72px_1fr_56px] items-center gap-3">
                <span className="text-sm capitalize text-ink-2">{r.t}</span>
                <div className="flex h-5 gap-[2px]" style={{ width: `${((r.confirm + r.override) / tierMax) * 100}%` }}>
                  {r.confirm > 0 && <div className="h-full rounded-l" style={{ flex: r.confirm, background: C.confirm }}
                    {...on(<><span className="font-medium capitalize">{r.t}</span> · {r.confirm} confirmed</>)} />}
                  {r.override > 0 && <div className="h-full rounded-r" style={{ flex: r.override, background: C.override }}
                    {...on(<><span className="font-medium capitalize">{r.t}</span> · {r.override} overridden</>)} />}
                </div>
                <span className="meta text-right">{r.confirm + r.override}</span>
              </div>
            ))}
          </div>
        </ChartCard>
        <ChartCard title="Override reasons" sub="Why the pharmacist disagreed">
          {reasons.length === 0 ? <div className="py-4 text-sm text-muted">No overrides yet.</div> : (
            <div className="flex flex-col gap-3">
              {reasons.map(([code, v]) => (
                <div key={code}>
                  <div className="mb-1 flex justify-between gap-3 text-sm"><span className="text-ink-2">{meta.override_reasons[code] ?? code}</span><span className="meta">{v}</span></div>
                  <div className="h-4 rounded-r" style={{ width: `${(v / reasonMax) * 100}%`, background: C.bar }}
                    {...on(<><span className="font-medium">{meta.override_reasons[code] ?? code}</span> · {v}</>)} />
                </div>
              ))}
            </div>
          )}
        </ChartCard>
      </div>
      <ChartCard title="Decisions per day" sub="Last 14 days, your local time">
        <div role="img" aria-label={`Decisions per day, last 14 days: ${perDay.join(', ')}`} className="flex h-36 items-end gap-2 border-b border-line">
          {perDay.map((v, i) => (
            <div key={i} className="flex h-full flex-1 items-end"
              {...on(<><span className="font-medium">{days[i].toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' })}</span> · {v} decision{v === 1 ? '' : 's'}</>)}>
              <div className="w-full rounded-t" style={{ height: v ? `${(v / dayMax) * 100}%` : 0, minHeight: v ? 3 : 0, background: C.bar }} />
            </div>
          ))}
        </div>
        <div className="mt-1.5 flex gap-2">
          {days.map((d, i) => <span key={i} className="meta flex-1 text-center text-[11px]">{i % 2 === 1 || i === 13 ? d.getDate() : ''}</span>)}
        </div>
      </ChartCard>
    </div>
  )
}
