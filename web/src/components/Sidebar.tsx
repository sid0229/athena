import { ArrowLeftRight, ChevronDown, FileText, Moon, Pill, Settings, ShieldCheck, Sun, User } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Reviewer } from '../types'
import { Button, ICON, cx } from './ui'

export type Page = 'input' | 'triage' | 'diff' | 'meds' | 'audit'

const NAV: { id: Page; label: string; icon: typeof FileText }[] = [
  { id: 'input', label: 'Input Note', icon: FileText },
  { id: 'triage', label: 'Triage', icon: ShieldCheck },
  { id: 'diff', label: 'Med Diff', icon: ArrowLeftRight },
  { id: 'meds', label: 'Medications', icon: Pill },
  { id: 'audit', label: 'Audit & System', icon: Settings },
]
export const ROLES = ['Clinical pharmacist', 'Clinical reviewer', 'Physician', 'Student reviewer']

export function Sidebar({ page, setPage, badges, version, reviewer, setReviewer, theme, toggleTheme, hasReport }: {
  page: Page; setPage: (p: Page) => void; badges: Partial<Record<Page, number>>; version: string
  reviewer: Reviewer; setReviewer: (r: Reviewer) => void; theme: 'light' | 'dark'; toggleTheme: () => void
  hasReport: boolean
}) {
  return (
    <aside className="fixed inset-y-0 left-0 z-20 flex w-[248px] flex-col border-r border-line bg-surface">
      <div className="flex h-[68px] items-center gap-2.5 border-b border-line px-5">
        <img src="/athena-mark.png" alt="" className="size-7 rounded dark:bg-white dark:p-0.5" />
        <span className="text-[16px] font-semibold tracking-tight text-ink">athena</span>
        <span className="meta">{version}</span>
      </div>
      <nav className="flex flex-col gap-0.5 py-3" aria-label="Main">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = page === id
          const disabled = id !== 'input' && id !== 'audit' && !hasReport
          return (
            <button key={id} onClick={() => setPage(id)} disabled={disabled}
              title={disabled ? 'Analyze a note first' : undefined}
              aria-current={active ? 'page' : undefined}
              className={cx('relative flex h-10 items-center gap-3 px-5 text-[14px] transition-colors cursor-pointer',
                'disabled:cursor-not-allowed disabled:opacity-40',
                active ? 'bg-accent-soft text-accent-text font-medium' : 'text-ink-2 hover:bg-surface-2 hover:text-ink')}>
              {active && <span className="absolute inset-y-0 left-0 w-[3px] bg-accent" />}
              <Icon {...ICON} />
              <span className="flex-1 text-left">{label}</span>
              {!!badges[id] && (
                <span className="min-w-5 rounded border border-line-strong px-1.5 text-center font-mono text-[11px] text-muted">
                  {badges[id]}
                </span>
              )}
            </button>
          )
        })}
      </nav>
      <div className="mt-auto border-t border-line px-4 py-4">
        <ReviewerMenu reviewer={reviewer} setReviewer={setReviewer} />
        <div className="mt-3 flex items-center justify-between px-1">
          <span className="eyebrow">Mode</span>
          <button onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="grid size-8 place-items-center rounded-md border border-line text-muted hover:text-ink cursor-pointer">
            {theme === 'dark' ? <Moon {...ICON} /> : <Sun {...ICON} className="text-review" />}
          </button>
        </div>
      </div>
    </aside>
  )
}

function ReviewerMenu({ reviewer, setReviewer }: { reviewer: Reviewer; setReviewer: (r: Reviewer) => void }) {
  const [open, setOpen] = useState(!reviewer.name)
  const [name, setName] = useState(reviewer.name)
  const [role, setRole] = useState(reviewer.role || ROLES[0])
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (open && reviewer.name && !ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [open, reviewer.name])
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(o => !o)} aria-expanded={open}
        className="flex w-full items-center gap-2.5 rounded-md px-1 py-1.5 text-left hover:bg-surface-2 cursor-pointer">
        <User {...ICON} className="text-muted" />
        <span className="flex-1 truncate text-[14px] text-ink">{reviewer.name || 'Set reviewer'}</span>
        <ChevronDown size={14} strokeWidth={1.75} className={cx('text-muted transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <form onSubmit={e => { e.preventDefault(); if (name.trim()) { setReviewer({ name: name.trim(), role }); setOpen(false) } }}
          className="absolute bottom-full left-0 right-0 mb-2 rounded-lg border border-line bg-surface p-3 shadow-lg">
          <div className="eyebrow mb-2">Reviewer</div>
          <input autoFocus value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Dr. A. Sharma"
            className="h-9 w-full rounded-md border border-line-strong bg-surface px-2.5 text-[13px] text-ink placeholder:text-faint" />
          <select value={role} onChange={e => setRole(e.target.value)}
            className="mt-2 h-9 w-full rounded-md border border-line-strong bg-surface px-2 text-[13px] text-ink">
            {ROLES.map(r => <option key={r}>{r}</option>)}
          </select>
          <p className="mt-2 text-[11.5px] leading-snug text-muted">Local session only. Your name is written to the audit log with each decision.</p>
          <Button kind="primary" type="submit" disabled={!name.trim()} className="mt-2 w-full">Save</Button>
        </form>
      )}
    </div>
  )
}
