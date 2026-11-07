import { Check, ChevronsRight, LoaderCircle } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { Decision, Tier } from '../types'
import { fmtTs } from '../lib'

export const ICON = { size: 16, strokeWidth: 1.75 } as const

export function cx(...c: (string | false | null | undefined)[]) { return c.filter(Boolean).join(' ') }

type BtnKind = 'primary' | 'outline' | 'ghost'
export function Button({ kind = 'outline', icon, children, className, ...p }:
  { kind?: BtnKind; icon?: ReactNode; children?: ReactNode } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...p} className={cx(
      'inline-flex items-center justify-center gap-1.5 rounded-md text-sm font-medium h-10 px-4 transition-colors',
      'disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap',
      kind === 'primary' && 'bg-accent text-accent-ink hover:opacity-90',
      kind === 'outline' && 'border border-line-strong bg-surface text-ink hover:bg-surface-2',
      kind === 'ghost' && 'text-muted hover:text-ink',
      className)}>
      {icon}{children}
    </button>
  )
}

export const Spinner = ({ size = 15 }: { size?: number }) => <LoaderCircle size={size} className="animate-spin" />

const DOT: Record<Tier, string> = {
  critical: 'bg-critical',
  review: 'border-2 border-review',
  info: 'border-2 border-info',
}
// Shape differs by tier too (filled vs ring), and the tier is announced, so meaning is not colour-only.
export const TierDot = ({ tier }: { tier: Tier }) =>
  <span role="img" aria-label={`${tier} tier`} className={cx('inline-block size-2.5 rounded-full shrink-0', DOT[tier])} />

export type Tone = 'critical' | 'review' | 'info' | 'ok' | 'hold' | 'neutral'
const TONE: Record<Tone, string> = {
  critical: 'text-critical bg-critical-soft border-critical-line',
  review: 'text-review bg-review-soft border-review-line',
  info: 'text-info bg-info-soft border-info-line',
  ok: 'text-ok bg-ok-soft border-ok-line',
  hold: 'text-hold bg-hold-soft border-hold-line',
  neutral: 'text-muted bg-surface-2 border-line',
}
export const Badge = ({ tone, children }: { tone: Tone; children: ReactNode }) =>
  <span className={cx('inline-flex items-center gap-1 rounded border px-1.5 py-px font-mono text-xs', TONE[tone])}>{children}</span>

export function Tabs<T extends string>({ value, onChange, items }:
  { value: T; onChange: (v: T) => void; items: { id: T; label: ReactNode; icon?: ReactNode }[] }) {
  return (
    <div role="group" className="flex items-center gap-6 border-b border-line">
      {items.map(it => (
        <button key={it.id} aria-pressed={value === it.id} onClick={() => onChange(it.id)}
          className={cx('-mb-px flex items-center gap-1.5 border-b-2 pb-3 pt-2 text-base cursor-pointer transition-colors',
            value === it.id ? 'border-ink text-ink font-medium' : 'border-transparent text-muted hover:text-ink')}>
          {it.icon}{it.label}
        </button>
      ))}
    </div>
  )
}

export const PageHeader = ({ eyebrow, title, tag, right }:
  { eyebrow?: ReactNode; title: string; tag?: ReactNode; right?: ReactNode }) => (
  <div className="mb-6 flex items-end justify-between gap-6">
    <div>
      {eyebrow && <div className="eyebrow mb-2">{eyebrow}</div>}
      <h1 className="flex items-baseline gap-3 text-2xl font-semibold tracking-tight text-ink">
        {title}{tag && <span className="meta">{tag}</span>}
      </h1>
    </div>
    {right && <div className="flex items-center gap-3">{right}</div>}
  </div>
)

export const Card = ({ children, className }: { children: ReactNode; className?: string }) =>
  <div className={cx('rounded-lg border border-line bg-surface', className)}>{children}</div>

export const Empty = ({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) => (
  <Card className="flex flex-col items-center px-6 py-14 text-center">
    <div className="mb-3 text-faint">{icon}</div>
    <div className="text-base font-medium text-ink">{title}</div>
    {children && <div className="mt-1.5 max-w-md text-sm text-muted">{children}</div>}
  </Card>
)

/** Confirm / Override actions with an inline override form (hook: returns buttons, status, panel). */
export function useDecision({ decision, reasons, reviewer, onDecide }: {
  decision?: Decision
  reasons: Record<string, string>
  reviewer: string
  onDecide: (action: 'confirm' | 'override', reason?: string, comment?: string) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [reason, setReason] = useState(Object.keys(reasons)[0] ?? 'other')
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const noReviewer = !reviewer.trim()
  const run = async (a: 'confirm' | 'override') => {
    setBusy(true); setError(null)
    try {
      await onDecide(a, a === 'override' ? reason : undefined, a === 'override' ? comment : undefined)
      setOpen(false); setComment('')
    } catch (e) {
      setError(`Not recorded: ${(e as Error).message}`)
    } finally { setBusy(false) }
  }
  return {
    buttons: (
      <div className="flex items-center gap-2" title={noReviewer ? 'Set your name in the sidebar first' : undefined}>
        {error && !open && <span role="alert" className="max-w-56 text-xs text-critical">{error}</span>}
        <Button disabled={noReviewer || busy} onClick={() => run('confirm')}
          icon={<Check size={15} strokeWidth={2} className="text-ok" />}
          className={decision?.action === 'confirm' ? 'border-ok-line bg-ok-soft' : ''}>
          {decision?.action === 'confirm' ? 'Confirmed' : 'Confirm'}
        </Button>
        <Button kind="ghost" disabled={noReviewer || busy} onClick={() => setOpen(o => !o)} aria-expanded={open}
          icon={<ChevronsRight size={15} strokeWidth={1.75} />}
          className={decision?.action === 'override' ? 'text-review' : ''}>
          {decision?.action === 'override' ? 'Overridden' : 'Override'}
        </Button>
      </div>
    ),
    status: decision ? (
      <span className="meta">
        {decision.action === 'confirm' ? 'Confirmed' : `Overridden · ${reasons[decision.reason_code ?? ''] ?? ''}`}
        {' · '}{decision.reviewer} · {fmtTs(decision.ts)}
      </span>
    ) : <span className="meta">Unreviewed</span>,
    panel: open ? (
      <div className="mt-3 rounded-md border border-line bg-surface-2 p-4">
        <div className="mb-3 text-sm font-medium text-ink">Override — a reason is required</div>
        <div className="grid gap-3 sm:grid-cols-[1fr_1.4fr]">
          <label className="block">
            <span className="eyebrow">Reason</span>
            <select value={reason} onChange={e => setReason(e.target.value)}
              className="mt-1 h-11 w-full rounded-md border border-line-strong bg-surface px-2 text-sm text-ink">
              {Object.entries(reasons).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <label className="block">
            <span className="eyebrow">Comment{reason === 'other' ? ' (required)' : ''}</span>
            <input value={comment} onChange={e => setComment(e.target.value)} placeholder="Clinical rationale…"
              className="mt-1 h-11 w-full rounded-md border border-line-strong bg-surface px-2.5 text-sm text-ink placeholder:text-faint" />
          </label>
        </div>
        <div className="mt-3 flex items-center justify-between gap-3">
          {error ? <span role="alert" className="text-xs text-critical">{error}</span>
            : <span className="meta">Recorded in the audit log with your name.</span>}
          <div className="flex gap-2">
            <Button kind="ghost" onClick={() => setOpen(false)}>Cancel</Button>
            <Button kind="primary" disabled={busy || (reason === 'other' && !comment.trim())}
              onClick={() => run('override')}>{busy ? <Spinner /> : null}Record override</Button>
          </div>
        </div>
      </div>
    ) : null,
  }
}
