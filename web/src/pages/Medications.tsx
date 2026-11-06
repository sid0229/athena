import { ChevronDown, ChevronUp, FileText, X } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import type { Analysis } from '../types'
import { cap, disp } from '../lib'
import { Button, Card, PageHeader, cx } from '../components/ui'

const STATUS: Record<string, string> = { active: 'text-ok', held: 'text-hold', stopped: 'text-critical', mentioned: 'text-muted' }
const CHECKED = new Set(['mapped'])

export function Medications({ a, analyzedAt }: { a: Analysis; analyzedAt: string }) {
  const [open, setOpen] = useState<string | null>(null)
  const [note, setNote] = useState(false)
  const meds = a.report.medications
  return (
    <div className="mx-auto max-w-[940px]">
      <PageHeader title="Medication Record" tag={<span className="rounded border border-line px-1.5 py-px">{meds.length} entries</span>}
        right={<Button icon={<FileText size={15} strokeWidth={1.75} />} onClick={() => setNote(true)}>View Source Note</Button>} />
      <Card className="overflow-hidden">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-line">
              {['Medication', 'Status', 'Stage', ''].map((h, i) => (
                <th key={i} className={cx('eyebrow px-4 py-3 font-normal', i === 3 && 'text-right')}>{h || 'Details'}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {meds.map(m => {
              const unchecked = !CHECKED.has(m.norm_status)
              const isOpen = open === m.id
              const stage = [m.in_admission && 'Adm', m.in_discharge && 'Dis'].filter(Boolean).join(' · ') || 'Narrative'
              const dose = [m.attributes.Strength, m.attributes.Route, m.attributes.Frequency].filter(Boolean).join(' · ')
              return (
                <Fragment key={m.id}>
                  <tr onClick={() => setOpen(isOpen ? null : m.id)}
                    className={cx('cursor-pointer border-b border-line last:border-b-0 hover:bg-surface-2', isOpen && unchecked && 'bg-critical-soft')}>
                    <td className="px-4 py-3.5">
                      <span className="text-[14.5px] text-ink">{disp(m.name, a.display_names)}</span>
                      {m.attributes.Strength && <span className="meta ml-2">{m.attributes.Strength}</span>}
                      {unchecked && <span className="ml-2 inline-block size-1.5 rounded-full bg-critical align-middle" aria-label="not checked" />}
                    </td>
                    <td className="px-4 py-3.5">
                      <span className={cx('flex items-center gap-1.5 text-[13.5px]', STATUS[m.status])}>
                        <span className="size-1.5 rounded-full bg-current" /><span className="text-ink">{cap(m.status)}</span>
                      </span>
                    </td>
                    <td className="meta px-4 py-3.5">{stage}</td>
                    <td className="px-4 py-3.5 text-right text-muted">
                      <button aria-expanded={isOpen} aria-label="Details" className="cursor-pointer">
                        {isOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                      </button>
                    </td>
                  </tr>
                  {isOpen && (
                    <tr className="border-b border-line"><td colSpan={4} className="px-4 pb-4 pt-1">
                      <div className={cx('rounded-md border p-4', unchecked ? 'border-critical-line' : 'border-line')}>
                        <div className="eyebrow mb-1.5">Raw extracted string</div>
                        <div className="meta rounded border border-line bg-surface-2 px-3 py-2 text-ink-2">
                          as written: “{m.raw}”{dose ? ` · ${dose}` : ''}{m.text_status ? ` · note says “${m.text_status}”` : ''}
                        </div>
                        <div className="mt-4 grid grid-cols-2 gap-4">
                          <div>
                            <div className="eyebrow mb-1.5">Mapping status</div>
                            {unchecked
                              ? <div className="flex items-center gap-1.5 text-[13.5px] text-critical"><span className="size-1.5 rounded-full bg-critical" />
                                {m.norm_status === 'drug_class' ? 'Drug class — interactions not checked'
                                  : m.norm_status === 'ambiguous' ? `Ambiguous (${m.kb_names.join(', ')}) — not checked`
                                    : 'Not mapped to a known drug — interactions not checked'}</div>
                              : <div className="flex items-center gap-1.5 text-[13.5px] text-ink"><span className="size-1.5 rounded-full bg-ok" />
                                Mapped to {m.kb_names.join(' + ')} <span className="meta">({m.norm_stage.replace('_', ' ')} match)</span></div>}
                          </div>
                          <div>
                            <div className="eyebrow mb-1.5">Confidence score</div>
                            <div className="text-[13.5px]"><span className="font-mono font-semibold text-ink">{m.confidence.toFixed(2)}</span>
                              <span className="meta"> / 1.00 · {Object.entries(m.confidence_parts).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ')}</span></div>
                          </div>
                        </div>
                        {unchecked && <div className="meta mt-4 border-t border-line pt-3">Action: review this medication manually — Athena could not check its interactions.</div>}
                      </div>
                    </td></tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </Card>
      <div className="mt-5 flex items-center justify-between">
        <span className="meta flex items-center gap-4">
          <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-ok" />Verified concept</span>
          <span className="flex items-center gap-1.5"><span className="size-1.5 rounded-full bg-critical" />Manual review needed</span>
        </span>
        <span className="meta">Analyzed {analyzedAt} · {a.report.meta.extraction_mode} · {a.report.meta.extraction_seconds}s</span>
      </div>
      {note && <NoteDrawer a={a} onClose={() => setNote(false)} />}
    </div>
  )
}

function NoteDrawer({ a, onClose }: { a: Analysis; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])
  const hot = new Set(a.report.findings.filter(f => f.tier === 'critical').flatMap(f => f.meds))
  const spans = a.report.medications.flatMap(m => m.mentions.map(x => ({ ...x, id: m.id, name: m.name })))
    .sort((x, y) => x.start - y.start)
  const parts: React.ReactNode[] = []
  let pos = 0
  spans.forEach((s, i) => {
    if (s.start < pos) return
    parts.push(a.text.slice(pos, s.start))
    parts.push(<mark key={i} className={cx('drug', hot.has(s.id) && 'hot')} title={disp(s.name, a.display_names)}>{a.text.slice(s.start, s.end)}</mark>)
    pos = s.end
  })
  parts.push(a.text.slice(pos))
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onClick={onClose}>
      <div role="dialog" aria-label="Source note" onClick={e => e.stopPropagation()}
        className="flex h-full w-full max-w-[680px] flex-col border-l border-line bg-surface">
        <div className="flex h-[68px] items-center justify-between border-b border-line px-5">
          <div><div className="text-[15px] font-semibold text-ink">Source note</div>
            <div className="meta mt-0.5 flex gap-3"><span><mark className="drug">drug</mark> extracted</span><span><mark className="drug hot">drug</mark> in a critical finding</span></div></div>
          <button onClick={onClose} aria-label="Close" className="text-muted hover:text-ink cursor-pointer"><X size={18} /></button>
        </div>
        <pre className="scroll-thin flex-1 overflow-auto whitespace-pre-wrap px-5 py-4 font-sans text-[13.5px] leading-[1.8] text-ink-2">{parts}</pre>
      </div>
    </div>
  )
}
