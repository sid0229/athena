import { ArrowLeftRight, ChevronDown, FileText, Moon, PanelLeftClose, PanelLeftOpen, Pill, Settings, ShieldCheck, Sun, User } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { Reviewer } from '../types'
import { initials } from '../lib'
import { Button, ICON, cx } from './ui'

export type Page = 'input' | 'triage' | 'diff' | 'meds' | 'audit'
export const SIDEBAR_W = { open: 248, closed: 68 }

const NAV: { id: Page; label: string; icon: typeof FileText }[] = [
  { id: 'input', label: 'Input Note', icon: FileText },
  { id: 'triage', label: 'Triage', icon: ShieldCheck },
  { id: 'diff', label: 'Med Diff', icon: ArrowLeftRight },
  { id: 'meds', label: 'Medications', icon: Pill },
  { id: 'audit', label: 'Audit & System', icon: Settings },
]
export const ROLES = ['Clinical pharmacist', 'Clinical reviewer', 'Physician', 'Student reviewer']
export const MOD = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform) ? '⌘' : 'Ctrl'

/** Label shown to the right of an icon when the sidebar is collapsed — on hover and on keyboard focus. */
function Tip({ show, children }: { show: boolean; children: ReactNode }) {
  if (!show) return null
  return (
    <span role="tooltip"
      className="pointer-events-none absolute left-full top-1/2 z-50 ml-3 -translate-y-1/2 whitespace-nowrap rounded-md bg-ink px-2.5 py-1.5 text-xs font-medium text-bg opacity-0 shadow-lg transition-opacity duration-100 group-hover:opacity-100 group-focus-visible:opacity-100">
      {children}
    </span>
  )
}

export function Sidebar({ page, setPage, badges, version, reviewer, setReviewer, theme, toggleTheme, hasReport, collapsed, setCollapsed }: {
  page: Page; setPage: (p: Page) => void; badges: Partial<Record<Page, number>>; version: string
  reviewer: Reviewer; setReviewer: (r: Reviewer) => void; theme: 'light' | 'dark'; toggleTheme: () => void
  hasReport: boolean; collapsed: boolean; setCollapsed: (c: boolean) => void
}) {
  const toggleLabel = `${collapsed ? 'Open' : 'Close'} sidebar (${MOD}+B)`
  return (
    <aside aria-label="Sidebar" style={{ width: collapsed ? SIDEBAR_W.closed : SIDEBAR_W.open }}
      className="fixed inset-y-0 left-0 z-20 flex flex-col border-r border-line bg-surface transition-[width] duration-200">
      <div className={cx('flex h-[68px] shrink-0 items-center border-b border-line', collapsed ? 'justify-center' : 'gap-2.5 pl-5 pr-3')}>
        {!collapsed && <>
          <img src="/athena-mark.png" alt="" className="size-7 rounded dark:bg-white dark:p-0.5" />
          <span className="text-lg font-semibold tracking-tight text-ink">athena</span>
          <span className="meta">{version}</span>
        </>}
        <button onClick={() => setCollapsed(!collapsed)} aria-label={toggleLabel} aria-expanded={!collapsed}
          className={cx('group relative grid size-10 place-items-center rounded-md text-muted hover:bg-surface-2 hover:text-ink cursor-pointer', !collapsed && 'ml-auto')}>
          {collapsed ? <>
            {/* Collapsed: the Athena mark; on hover / keyboard focus it turns into the open-sidebar icon */}
            <img src="/athena-mark.png" alt="" className="size-7 rounded group-hover:hidden group-focus-visible:hidden dark:bg-white dark:p-0.5" />
            <PanelLeftOpen {...ICON} size={18} aria-hidden className="hidden group-hover:block group-focus-visible:block" />
          </> : <PanelLeftClose {...ICON} size={18} aria-hidden />}
          <Tip show={collapsed}>{toggleLabel}</Tip>
        </button>
      </div>

      <nav className="flex flex-col gap-1 py-3" aria-label="Main">
        {NAV.map(({ id, label, icon: Icon }) => {
          const active = page === id
          const disabled = id !== 'input' && id !== 'audit' && !hasReport
          const n = badges[id]
          return (
            <button key={id} onClick={() => setPage(id)} disabled={disabled}
              aria-label={collapsed ? `${label}${n ? `, ${n} pending` : ''}` : undefined}
              aria-current={active ? 'page' : undefined}
              title={disabled && !collapsed ? 'Analyze a note first' : undefined}
              className={cx('group relative flex h-11 items-center text-base transition-colors cursor-pointer',
                collapsed ? 'mx-auto w-11 justify-center rounded-md' : 'gap-3 px-5',
                'disabled:cursor-not-allowed disabled:opacity-40',
                active ? 'bg-accent-soft font-medium text-accent-text' : 'text-ink-2 hover:bg-surface-2 hover:text-ink')}>
              {active && !collapsed && <span className="absolute inset-y-0 left-0 w-[3px] bg-accent" />}
              <Icon {...ICON} size={18} aria-hidden />
              {!collapsed && <span className="flex-1 text-left">{label}</span>}
              {!!n && (collapsed
                ? <span aria-hidden className="absolute right-0.5 top-0.5 min-w-4 rounded-full bg-accent px-1 text-center font-mono text-[10px] leading-4 text-accent-ink">{n}</span>
                : <span className="min-w-6 rounded border border-line-strong px-1.5 text-center font-mono text-xs text-muted">{n}</span>)}
              <Tip show={collapsed}>{label}{n ? ` · ${n} pending` : ''}{disabled ? ' · analyze a note first' : ''}</Tip>
            </button>
          )
        })}
      </nav>

      <div className={cx('mt-auto border-t border-line py-4', collapsed ? 'flex flex-col items-center gap-2' : 'px-4')}>
        <ReviewerMenu reviewer={reviewer} setReviewer={setReviewer} collapsed={collapsed} />
        <div className={cx('flex items-center', collapsed ? '' : 'mt-3 justify-between px-1')}>
          {!collapsed && <span className="eyebrow">Theme</span>}
          <button onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            className="group relative grid size-10 place-items-center rounded-md border border-line text-muted hover:text-ink cursor-pointer">
            {theme === 'dark' ? <Moon {...ICON} aria-hidden /> : <Sun {...ICON} className="text-review" aria-hidden />}
            <Tip show={collapsed}>{theme === 'dark' ? 'Dark mode — switch to light' : 'Light mode — switch to dark'}</Tip>
          </button>
        </div>
      </div>
    </aside>
  )
}

function ReviewerMenu({ reviewer, setReviewer, collapsed }: { reviewer: Reviewer; setReviewer: (r: Reviewer) => void; collapsed: boolean }) {
  const [open, setOpen] = useState(!reviewer.name)
  const [name, setName] = useState(reviewer.name)
  const [role, setRole] = useState(reviewer.role || ROLES[0])
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const close = (e: MouseEvent) => { if (open && reviewer.name && !ref.current?.contains(e.target as Node)) setOpen(false) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape' && reviewer.name) setOpen(false) }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', esc)
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc) }
  }, [open, reviewer.name])
  const label = reviewer.name ? `${reviewer.name} · ${reviewer.role || 'Reviewer'}` : 'Set reviewer'
  return (
    <div ref={ref} className={cx('relative', !collapsed && 'w-full')}>
      <button onClick={() => setOpen(o => !o)} aria-expanded={open} aria-haspopup="dialog"
        aria-label={collapsed ? `Reviewer: ${label}` : undefined}
        className={cx('group relative flex items-center rounded-md hover:bg-surface-2 cursor-pointer',
          collapsed ? 'size-10 justify-center' : 'h-12 w-full gap-2.5 px-1.5 text-left')}>
        {collapsed
          ? <span className="grid size-8 place-items-center rounded-full bg-accent text-xs font-semibold text-accent-ink">{reviewer.name ? initials(reviewer.name) : <User size={15} aria-hidden />}</span>
          : <User {...ICON} className="text-muted" aria-hidden />}
        {!collapsed && <>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-base text-ink">{reviewer.name || 'Set reviewer'}</span>
            {reviewer.name && <span className="block truncate text-xs text-muted">{reviewer.role}</span>}
          </span>
          <ChevronDown size={16} strokeWidth={1.75} aria-hidden className={cx('text-muted transition-transform', open && 'rotate-180')} />
        </>}
        <Tip show={collapsed && !open}>{label}</Tip>
      </button>
      {open && (
        <form role="dialog" aria-label="Reviewer"
          onSubmit={e => { e.preventDefault(); if (name.trim()) { setReviewer({ name: name.trim(), role }); setOpen(false) } }}
          className={cx('absolute z-50 rounded-lg border border-line bg-surface p-4 shadow-lg',
            collapsed ? 'bottom-0 left-full ml-3 w-72' : 'bottom-full left-0 right-0 mb-2')}>
          <label className="block">
            <span className="eyebrow">Reviewer name</span>
            <input autoFocus value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Dr. A. Sharma"
              className="mt-1.5 h-11 w-full rounded-md border border-line-strong bg-surface px-3 text-sm text-ink placeholder:text-faint" />
          </label>
          <label className="mt-3 block">
            <span className="eyebrow">Role</span>
            <select value={role} onChange={e => setRole(e.target.value)}
              className="mt-1.5 h-11 w-full rounded-md border border-line-strong bg-surface px-2.5 text-sm text-ink">
              {ROLES.map(r => <option key={r}>{r}</option>)}
            </select>
          </label>
          <p className="mt-3 text-xs text-muted">Local session only. Your name is written to the audit log with each decision.</p>
          <Button kind="primary" type="submit" disabled={!name.trim()} className="mt-3 w-full">Save</Button>
        </form>
      )}
    </div>
  )
}
