import { CircleCheck, Minus, Plus, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import type { Analysis, Finding, Meta, Tier } from '../types'
import { ACTIONABLE, TRIAGE_TYPES, disp, findingSummary, findingTitle, minConfidence } from '../lib'
import { Badge, Card, Empty, PageHeader, TierDot, cx, useDecision } from '../components/ui'

type Filter = 'all' | Tier

export function Triage({ a, meta, reviewer, onDecide }: {
  a: Analysis; meta: Meta; reviewer: string
  onDecide: (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => Promise<void>
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const items = a.report.findings.filter(f => TRIAGE_TYPES.has(f.type)).sort((x, y) => y.risk - x.risk)
  const count = (t: Filter) => items.filter(f => (t === 'all' ? ACTIONABLE(f) : f.tier === t)).length
  const shown = items.filter(f => (filter === 'all' ? ACTIONABLE(f) : f.tier === filter))
  const actionable = items.filter(ACTIONABLE)
  const reviewed = actionable.filter(f => a.decisions[f.id]).length
  const unchecked = a.report.findings.filter(f => f.type === 'NOT_CHECKED' || f.type === 'NOT_COVERED')

  return (
    <div className="mx-auto max-w-[1320px]">
      <PageHeader title="Interaction Triage" tag={a.case_id}
        right={<span className="meta">Reviewed <span className="text-ink-2">{reviewed} / {actionable.length}</span> · DrugBank + DDInter</span>} />
      <div className="mb-5 flex items-center justify-between border-b border-line">
        <div role="group" aria-label="Filter findings" className="flex items-center gap-1">
          {([['all', 'Needs review'], ['critical', 'Critical'], ['review', 'Review'], ['info', 'Info']] as [Filter, string][]).map(([id, label], i) => (
            <div key={id} className="flex items-center">
              {i > 0 && <span className="mx-2 text-faint">·</span>}
              <button aria-pressed={filter === id} onClick={() => setFilter(id)}
                className={cx('-mb-px border-b-2 pb-3 pt-2 text-base cursor-pointer',
                  filter === id ? 'border-ink font-medium text-ink' : 'border-transparent text-muted hover:text-ink')}>
                {label} ({count(id)})
              </button>
            </div>
          ))}
        </div>
        <span className="meta pb-2.5">Sort: risk score</span>
      </div>

      {shown.length === 0 ? (
        <Empty icon={<CircleCheck size={24} strokeWidth={1.5} />} title={filter === 'all' ? 'Nothing needs review' : `No ${filter} items`}>
          {filter === 'all' ? 'No critical or review interactions. Lower-priority records are under Info.' : null}
        </Empty>
      ) : (
        <div className="flex flex-col gap-3">
          {shown.map(f => <FindingCard key={f.id} f={f} a={a} meta={meta} reviewer={reviewer} onDecide={onDecide} />)}
        </div>
      )}

      {unchecked.length > 0 && (
        <div className="mt-6 flex flex-col gap-2">
          {unchecked.map(f => (
            <div key={f.id} className="flex items-start gap-2.5 rounded-md border border-line bg-surface px-4 py-3 text-sm text-muted">
              <TriangleAlert size={15} strokeWidth={1.75} className="mt-0.5 shrink-0 text-hold" />
              <div className="min-w-0 flex-1"><span className="font-medium text-ink">{f.title}:</span> {f.explanation.slice(0, -1).join('; ')}.
                <span className="block text-faint">{f.explanation.slice(-1)[0]}</span>
                {f.type === 'NOT_COVERED' && !!a.ml_signals?.length && (
                  <div className="mt-3 rounded-md border border-line bg-surface-2 px-3 py-2.5">
                    <div className="eyebrow mb-1.5">Research model signal — not verified, not used for flags</div>
                    <table className="w-full text-sm">
                      <tbody>
                        {a.ml_signals.map(s => (
                          <tr key={`${s.a}|${s.b}`} className="border-t border-line first:border-t-0">
                            <td className="py-1 pr-3 text-ink-2">{disp(s.a_name, a.display_names)} + {disp(s.b_name, a.display_names)}</td>
                            <td className="py-1 pr-3 text-right font-mono">{(s.probability * 100).toFixed(0)}%</td>
                            <td className="meta py-1 text-right">{s.model}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-1.5 text-xs text-muted">Probability that DrugBank would record an interaction, from the XGBoost predictor (benchmark ROC-AUC 0.98 for known drugs, 0.71–0.81 for new ones). Check a drug reference before acting.</p>
                  </div>
                )}</div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-8 flex items-center justify-between border-t border-line pt-4">
        <span className="meta">{shown.length} of {items.length} interaction records shown · flags come from DrugBank and DDInter, never the LLM</span>
        <span className="meta">Athena {meta.version}</span>
      </div>
    </div>
  )
}

function FindingCard({ f, a, meta, reviewer, onDecide }: {
  f: Finding; a: Analysis; meta: Meta; reviewer: string
  onDecide: (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const ev = f.evidence
  const conf = minConfidence(f, a.report.medications)
  const d = useDecision({ decision: a.decisions[f.id], reasons: meta.override_reasons, reviewer,
    onDecide: (act, r, c) => onDecide(f, act, r, c) })
  const riskLine = f.explanation.find(x => x.startsWith('Risk '))
  const names = Object.fromEntries(a.report.medications.map(m => [m.id, disp(m.name, a.display_names)]))

  return (
    <Card className={cx('px-4 pt-4', open && 'ring-1 ring-line-strong')}>
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2.5">
            <TierDot tier={f.tier} />
            <h3 className="truncate text-lg font-semibold tracking-tight text-ink">{findingTitle(f, a)}</h3>
            {ev.severity && <Badge tone={ev.severity === 'Major' ? 'critical' : ev.severity === 'Moderate' ? 'review' : 'neutral'}>
              {ev.severity}{ev.severity_basis === 'derived' ? ' · derived' : ev.severity_basis === 'rule' ? ' · rule' : ''}</Badge>}
            {f.type === 'DUPLICATE_THERAPY' && <Badge tone="review">Duplicate therapy</Badge>}
          </div>
          <p className={cx('mt-1 pl-5 text-sm text-ink-2', !open && 'truncate')}>{findingSummary(f)}</p>
          {f.needs_verification && (
            <p className="mt-1.5 flex items-center gap-1.5 pl-5 text-xs text-review">
              <TriangleAlert size={13} strokeWidth={2} />Possible Major interaction — verify the extracted drugs
            </p>
          )}
        </div>
        {d.buttons}
      </div>
      {d.panel}
      <div className="mt-3 flex items-center justify-between border-t border-line py-2.5">
        <button onClick={() => setOpen(o => !o)} aria-expanded={open}
          className="meta flex items-center gap-1.5 hover:text-ink cursor-pointer">
          {open ? <Minus size={12} /> : <Plus size={12} />}Evidence &amp; risk score
        </button>
        {open ? d.status : <span className="meta">Score: <span className="text-ink-2">{f.risk.toFixed(2)}</span></span>}
      </div>
      {open && (
        <div className="border-t border-line pb-4 pt-4">
          <div className="grid grid-cols-3 gap-4">
            <div><div className="meta">Risk score</div>
              <div className={cx('mt-1 font-mono text-sm font-semibold', f.tier === 'critical' ? 'text-critical' : f.tier === 'review' ? 'text-review' : 'text-ink')}>{f.risk.toFixed(2)}</div></div>
            <div><div className="meta">Extraction confidence</div><div className="mt-1 font-mono text-sm text-ink">{conf?.toFixed(2) ?? '—'}</div></div>
            <div><div className="meta">Sources</div><div className="mt-1 text-sm text-ink">
              {(ev.sources ?? []).map(s => (s === 'ddinter' ? 'DDInter' : s === 'rule' ? 'High-alert rule' : 'DrugBank')).join(' · ') || '—'}</div></div>
          </div>
          {ev.rule && (
            <div className="mt-5 rounded-md border border-critical-line bg-critical-soft px-3.5 py-3 text-sm text-ink-2">
              <span className="font-medium text-critical">High-alert rule — {ev.rule.name}.</span> {ev.rule.effect}.
              <span className="mt-1 block text-xs text-muted">Source: {ev.rule.source}</span>
            </div>
          )}
          {(ev.ddinter_level || ev.mechanisms?.length) ? (
            <>
              <div className="eyebrow mt-5 mb-2">Evidence</div>
              <div className="rounded-md border border-line bg-surface-2 px-3.5 py-3 text-sm leading-relaxed text-ink-2">
                {ev.ddinter_level && <p>DDInter grade: <span className="font-medium text-ink">{ev.ddinter_level}</span></p>}
                {ev.mechanisms?.map((m, i) => <p key={i} className={ev.ddinter_level || i ? 'mt-1.5' : ''}>DrugBank: {m.text}</p>)}
                {ev.severity_basis === 'derived' && ev.mechanisms?.[0] && (
                  <p className="mt-1.5 text-xs text-muted">Severity derived: {ev.mechanisms[0].pct_major.toFixed(0)}% of {ev.mechanisms[0].n_overlap_graded} comparable DrugBank pairs are graded Major by DDInter.</p>
                )}
              </div>
            </>
          ) : null}
          {ev.confidence && (
            <div className="mt-4 grid gap-1">
              {Object.entries(ev.confidence).map(([mid, parts]) => (
                <div key={mid} className="meta flex justify-between gap-4">
                  <span className="text-ink-2">{names[mid] ?? mid}</span>
                  <span>{Object.entries(parts).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ')}</span>
                </div>
              ))}
            </div>
          )}
          {riskLine && <div className="meta mt-4 border-t border-line pt-3">{riskLine}</div>}
        </div>
      )}
    </Card>
  )
}
