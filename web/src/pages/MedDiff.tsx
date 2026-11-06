import { Check, ChevronDown, ChevronUp, CircleMinus, Info, List, Plus, RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { Analysis, Finding, Medication, Meta } from '../types'
import { disp, lineAt } from '../lib'
import { Badge, Button, Card, Empty, PageHeader, Spinner, type Tone, cx, useDecision } from '../components/ui'

type Kind = 'omission' | 'held' | 'dose' | 'new' | 'same'
type Filter = 'all' | Kind
const KIND: Record<Kind, { label: string; tone: Tone; order: number }> = {
  omission: { label: 'Omission', tone: 'critical', order: 0 },
  dose: { label: 'Dose change', tone: 'info', order: 1 },
  held: { label: 'Stopped / held', tone: 'hold', order: 2 },
  new: { label: 'New', tone: 'info', order: 3 },
  same: { label: 'Unchanged', tone: 'ok', order: 4 },
}

interface Row { m: Medication; kind: Kind; finding?: Finding; adm: string; dis: string }

export function MedDiff({ a, meta, reviewer, busy, onDecide, onPriorList }: {
  a: Analysis; meta: Meta; reviewer: string; busy: boolean
  onDecide: (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => Promise<void>
  onPriorList: (prior: string) => void
}) {
  const kinds = new Set(a.report.meta.sections.map(s => s.kind))
  const byId = Object.fromEntries(a.report.findings.map(f => [f.id, f]))
  const rows: Row[] = useMemo(() => a.report.medications.filter(m => m.in_admission || m.in_discharge).map((m): Row => {
    const admS = m.admission_attributes.Strength || (m.in_admission ? 'dose not stated' : '—')
    const disS = m.attributes.Strength || 'dose not stated'
    if (m.in_admission && !m.in_discharge) {
      const f = byId[`OMIT:${m.id}`]
      return { m, kind: f?.tier === 'info' ? 'held' : 'omission', finding: f, adm: admS, dis: f?.tier === 'info' ? 'Held' : 'Omitted' }
    }
    if (m.in_discharge && !m.in_admission) return { m, kind: 'new', finding: byId[`NEW:${m.id}`], adm: '—', dis: disS }
    const f = byId[`DOSE:${m.id}`]
    return { m, kind: f ? 'dose' : 'same', finding: f, adm: admS, dis: disS }
  }).sort((x, y) => KIND[x.kind].order - KIND[y.kind].order || x.m.name.localeCompare(y.m.name)),
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [a])
  const [filter, setFilter] = useState<Filter>('all')
  const [open, setOpen] = useState<string | null>(rows.find(r => r.kind === 'omission' || r.kind === 'dose')?.m.id ?? null)
  const [cursor, setCursor] = useState(0)
  const shown = rows.filter(r => filter === 'all' || r.kind === filter)

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input,textarea,select')) return
      if (e.key === 'j') setCursor(c => Math.min(c + 1, shown.length - 1))
      else if (e.key === 'k') setCursor(c => Math.max(c - 1, 0))
      else if (e.key === 'Enter' && shown[cursor]) setOpen(o => (o === shown[cursor].m.id ? null : shown[cursor].m.id))
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [shown, cursor])

  if (kinds.has('discharge') && !kinds.has('admission')) return <PriorListSetup a={a} busy={busy} onPriorList={onPriorList} />
  if (!(kinds.has('admission') && kinds.has('discharge'))) return (
    <div className="mx-auto max-w-[940px]">
      <PageHeader eyebrow="Reconciliation audit · admission → discharge" title="Regimen Differential" />
      <Empty icon={<Info size={24} strokeWidth={1.5} />} title="No discharge medication list found">
        Athena treated every medication in the note as active and checked them for interactions.
        Reconciliation needs a discharge medication list.
      </Empty>
    </div>
  )
  const flagged = rows.filter(r => r.kind === 'omission' || r.kind === 'dose').length
  const n = (k: Kind) => rows.filter(r => r.kind === k).length
  const pills: [Filter, string, typeof List][] = [['all', 'All Changes', List], ['omission', `Omissions (${n('omission')})`, CircleMinus],
    ['dose', `Dose Changes (${n('dose')})`, RefreshCw], ['new', `New (${n('new')})`, Plus], ['held', `Stopped / held (${n('held')})`, CircleMinus],
    ['same', `Unchanged (${n('same')})`, Check]]

  return (
    <div className="mx-auto max-w-[940px]">
      <PageHeader eyebrow={<>Reconciliation audit <span className="text-faint">/</span> Admission → Discharge</>}
        title="Regimen Differential"
        right={<div className="text-right">
          {flagged > 0 && <div className="meta flex items-center justify-end gap-1.5 text-critical"><span className="size-1.5 rounded-full bg-critical" />{flagged} flagged discrepanc{flagged === 1 ? 'y' : 'ies'}</div>}
          <div className="meta mt-1">{a.case_id}</div>
        </div>} />
      <div className="mb-4 flex flex-wrap items-center gap-1 border-b border-line pb-3">
        {pills.map(([id, label, Icon]) => (
          <button key={id} onClick={() => { setFilter(id); setCursor(0) }}
            className={cx('flex h-8 items-center gap-1.5 rounded-md px-3 text-[13.5px] cursor-pointer',
              filter === id ? 'border border-ink text-ink' : 'border border-transparent text-muted hover:text-ink')}>
            <Icon size={14} strokeWidth={1.75} />{label}
          </button>
        ))}
      </div>
      <Card className="overflow-hidden">
        {shown.length === 0 && <div className="px-4 py-10 text-center text-[13px] text-muted">No medications in this group.</div>}
        {shown.map((r, i) => (
          <DiffRow key={r.m.id} r={r} a={a} meta={meta} reviewer={reviewer} onDecide={onDecide}
            open={open === r.m.id} focused={cursor === i}
            toggle={() => { setOpen(o => (o === r.m.id ? null : r.m.id)); setCursor(i) }} />
        ))}
      </Card>
      <div className="mt-5 flex items-center justify-between">
        <span className="meta">{rows.length} medications evaluated · Reviewer: {reviewer || 'not set'}</span>
        <span className="meta flex items-center gap-1.5">
          <Kbd>J</Kbd><Kbd>K</Kbd> navigate <span className="mx-1 text-faint">|</span><Kbd>Enter</Kbd> toggle
        </span>
      </div>
    </div>
  )
}

const Kbd = ({ children }: { children: string }) =>
  <kbd className="rounded border border-line-strong bg-surface px-1.5 py-px font-mono text-[10.5px] text-ink-2">{children}</kbd>

function DiffRow({ r, a, meta, reviewer, onDecide, open, focused, toggle }: {
  r: Row; a: Analysis; meta: Meta; reviewer: string; open: boolean; focused: boolean; toggle: () => void
  onDecide: (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => Promise<void>
}) {
  const k = KIND[r.kind]
  const flagged = r.kind === 'omission' || r.kind === 'dose'
  const f = r.finding
  const d = useDecision({ decision: f ? a.decisions[f.id] : undefined, reasons: meta.override_reasons, reviewer,
    onDecide: (act, rs, c) => (f ? onDecide(f, act, rs, c) : Promise.resolve()) })
  const mention = (sec: string) => r.m.mentions.find(x => x.section === sec)
  const admLine = mention('admission'), disLine = mention('discharge')
  return (
    <div className={cx('border-b border-line last:border-b-0', focused && 'bg-surface-2')}>
      <button onClick={toggle} aria-expanded={open} className="flex h-14 w-full items-center gap-3 px-4 text-left cursor-pointer">
        <span className={cx('size-1.5 shrink-0 rounded-full', flagged ? 'bg-critical' : r.kind === 'same' ? 'bg-ok' : 'bg-transparent')} />
        <span className="flex-1 truncate text-[16px] font-medium text-ink">{disp(r.m.name, a.display_names)}</span>
        <span className="meta">{r.adm} → {r.dis}</span>
        <Badge tone={k.tone}>{k.label}</Badge>
        {a.decisions[f?.id ?? ''] && <Check size={14} strokeWidth={2.2} className="text-ok" aria-label="reviewed" />}
        {open ? <ChevronUp size={16} className="text-muted" /> : <ChevronDown size={16} className="text-muted" />}
      </button>
      {open && (
        <div className="border-t border-line px-4 pb-4 pt-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <Side title="Admission record" tag={r.m.in_admission ? 'On admission list' : 'Not listed'} tone={r.m.in_admission ? 'ok' : 'neutral'}
              text={admLine ? lineAt(a.text, admLine.start) : 'Not on the admission medication list.'} />
            <Side title="Discharge plan" tag={r.m.in_discharge ? (flagged ? 'Variance' : 'On discharge list') : 'Not listed'}
              tone={flagged ? 'critical' : r.m.in_discharge ? 'ok' : 'neutral'}
              text={disLine ? lineAt(a.text, disLine.start) : 'Not on the discharge medication list.'} />
          </div>
          {f && (
            <div className="mt-4 border-t border-line pt-4">
              <div className="flex items-center gap-3">
                <div className={cx('flex flex-1 items-start gap-2 rounded-md border px-3 py-2.5 text-[13.5px]',
                  flagged ? 'border-critical-line bg-critical-soft text-critical' : 'border-line bg-surface-2 text-ink-2')}>
                  {flagged ? <TriangleAlert size={15} strokeWidth={1.75} className="mt-0.5 shrink-0" /> : <Info size={15} strokeWidth={1.75} className="mt-0.5 shrink-0" />}
                  <span>{f.explanation.join(' ')}</span>
                </div>
                {d.buttons}
              </div>
              {d.panel}
              <div className="mt-2 text-right">{d.status}</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const Side = ({ title, tag, tone, text }: { title: string; tag: string; tone: Tone; text: string }) => (
  <div>
    <div className="mb-1.5 flex items-center justify-between">
      <span className="eyebrow">{title}</span>
      <span className={cx('meta', tone === 'ok' && 'text-ok', tone === 'critical' && 'text-critical')}>{tag}</span>
    </div>
    <div className="min-h-14 rounded-md border border-line px-3 py-2.5 text-[13px] leading-relaxed text-ink-2">“{text}”</div>
  </div>
)

function PriorListSetup({ a, busy, onPriorList }: { a: Analysis; busy: boolean; onPriorList: (p: string) => void }) {
  const [prior, setPrior] = useState('')
  const nDis = a.report.medications.filter(m => m.in_discharge).length
  return (
    <div className="mx-auto max-w-[940px]">
      <PageHeader eyebrow="Reconciliation audit · admission → discharge" title="Regimen Differential" />
      <Card className="p-6">
        <div className="flex items-baseline gap-2"><span className="text-[28px] font-semibold text-ink">{nDis}</span>
          <span className="text-[14px] text-muted">discharge medication{nDis === 1 ? '' : 's'} extracted and checked for interactions</span></div>
        <p className="mt-3 max-w-xl text-[13.5px] text-ink-2">No prior (admission) medication list was found in this note. Add one to find omissions,
          new medications and dose changes. Athena re-analyzes the note together with the list, on this device.</p>
        <textarea value={prior} onChange={e => setPrior(e.target.value)} rows={6}
          placeholder={'One medication per line, e.g.\nMetoprolol succinate 50 mg PO daily\nLisinopril 20 mg PO daily'}
          className="mt-4 w-full rounded-md border border-line-strong bg-surface px-3 py-2.5 text-[13.5px] leading-relaxed text-ink placeholder:text-faint" />
        <div className="mt-3 flex justify-end">
          <Button kind="primary" disabled={!prior.trim() || busy} onClick={() => onPriorList(prior)}
            icon={busy ? <Spinner /> : <RefreshCw size={15} strokeWidth={1.75} />}>Reconcile with this list</Button>
        </div>
      </Card>
    </div>
  )
}
