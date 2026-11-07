import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { SIDEBAR_W, Sidebar, type Page } from './components/Sidebar'
import { Spinner } from './components/ui'
import { ACTIONABLE, RECON_TYPES, TRIAGE_TYPES, initials } from './lib'
import { AuditSystem } from './pages/AuditSystem'
import { InputNote } from './pages/InputNote'
import { MedDiff } from './pages/MedDiff'
import { Medications } from './pages/Medications'
import { Triage } from './pages/Triage'
import type { Analysis, Finding, Meta, Reviewer } from './types'

function stored<T>(key: string, fallback: T): T {
  try { const v = localStorage.getItem(key); return v ? (JSON.parse(v) as T) : fallback } catch { return fallback }
}
function store(key: string, v: unknown) { try { localStorage.setItem(key, JSON.stringify(v)) } catch { /* private mode */ } }

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [metaError, setMetaError] = useState<string | null>(null)
  const [page, setPage] = useState<Page>('input')
  const pageRef = useRef(page)
  useEffect(() => { pageRef.current = page }, [page])
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (document.documentElement.dataset.theme as 'light' | 'dark') || 'light')
  const [reviewer, setReviewerState] = useState<Reviewer>(() => stored('athena-reviewer', { name: '', role: '' }))
  const [mode, setModeState] = useState<string>(() => stored('athena-mode', 'hybrid'))
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [note, setNote] = useState<{ text: string; label: string; source: string; at: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [llmOnline, setLlmOnline] = useState(false)
  const [auditKey, setAuditKey] = useState(0)
  const [collapsed, setCollapsedState] = useState<boolean>(() => stored('athena-sidebar-collapsed', false))
  const setCollapsed = (c: boolean) => { setCollapsedState(c); store('athena-sidebar-collapsed', c) }

  useEffect(() => { api.meta().then(m => { setMeta(m); setLlmOnline(m.llm_online) }).catch(e => setMetaError(String(e.message ?? e))) }, [])
  useEffect(() => {
    const id = setInterval(() => api.status().then(s => setLlmOnline(s.llm_online)).catch(() => {}), 15000)
    return () => clearInterval(id)
  }, [])
  useEffect(() => {  // Cmd/Ctrl + B toggles the sidebar, as in Claude
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.isContentEditable || t.closest('input,textarea,select')) return  // Ctrl+B moves the cursor in macOS fields
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        setCollapsedState(c => { store('athena-sidebar-collapsed', !c); return !c })
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('athena-theme', theme) } catch { /* ignore */ }
  }, [theme])

  const setReviewer = (r: Reviewer) => { setReviewerState(r); store('athena-reviewer', r) }
  const setMode = (m: string) => { setModeState(m); store('athena-mode', m) }

  const runAnalysis = useCallback(async (text: string, label: string, source: string, prior?: string) => {
    setBusy(true); setError(null)
    try {
      const a = await api.analyze(text, mode, prior)
      setAnalysis(a)
      setNote({ text, label, source, at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })
      if (!prior && pageRef.current === 'input') setPage('triage')  // don't pull the user off another page
    } catch (e) {
      setError(`Analysis failed: ${(e as Error).message}`)
    } finally { setBusy(false) }
  }, [mode])

  const decide = useCallback(async (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => {
    if (!analysis) return
    const hash = analysis.report.meta.report_hash
    // Throws on failure; the caller shows the error next to the buttons.
    const r = await api.decide({ report_hash: hash, finding_id: f.id, reviewer: reviewer.name, action,
      reason_code: reason ?? null, comment: comment ?? null })
    // Functional update guarded by the report hash: a late response never touches a newer analysis.
    setAnalysis(cur => (cur && cur.report.meta.report_hash === hash ? { ...cur, decisions: r.decisions } : cur))
    setAuditKey(k => k + 1)
  }, [analysis, reviewer.name])

  if (metaError) return (
    <div className="grid h-full place-items-center p-8 text-center">
      <div><div className="text-lg font-semibold text-ink">Cannot reach the local Athena server</div>
        <p className="mt-2 text-sm text-muted">Start it with <code className="font-mono">python app/server.py</code>, then reload. ({metaError})</p></div>
    </div>
  )
  if (!meta) return <div className="grid h-full place-items-center text-muted"><Spinner size={20} /></div>

  const pending = (types: Set<string>) =>
    analysis ? analysis.report.findings.filter(f => types.has(f.type) && ACTIONABLE(f) && !analysis.decisions[f.id]).length : 0

  return (
    <div className="min-h-full">
      <a href="#main" className="skip-link">Skip to content</a>
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} page={page} setPage={setPage} version={meta.version} reviewer={reviewer} setReviewer={setReviewer}
        theme={theme} toggleTheme={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))} hasReport={!!analysis}
        badges={{ triage: pending(TRIAGE_TYPES), diff: pending(RECON_TYPES) }} />
      <div className="transition-[padding] duration-200" style={{ paddingLeft: collapsed ? SIDEBAR_W.closed : SIDEBAR_W.open }}>
        <header className="sticky top-0 z-10 flex h-[68px] items-center justify-between border-b border-line bg-surface/90 px-6 backdrop-blur lg:px-8">
          <div className="meta truncate">
            {note && page !== 'input' && page !== 'audit' ? <>{note.source} · <span className="text-ink-2">{note.label}</span> · analyzed {note.at}</> : null}
          </div>
          <div role="img" aria-label={reviewer.name ? `Reviewer ${reviewer.name}` : 'No reviewer set'}
            title={reviewer.name ? `${reviewer.name} · ${reviewer.role}` : 'No reviewer set'}
            className="grid size-10 place-items-center rounded-full bg-accent text-sm font-semibold text-accent-ink">
            {reviewer.name ? initials(reviewer.name) : '?'}
          </div>
        </header>
        <main id="main" tabIndex={-1} className="px-6 py-8 outline-none lg:px-8">
          {error && page !== 'input' && (
            <div role="alert" className="mx-auto mb-6 flex max-w-[1320px] items-start justify-between gap-4 rounded-md border border-critical-line bg-critical-soft px-4 py-3 text-sm text-critical">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="font-medium underline cursor-pointer">Dismiss</button>
            </div>
          )}
          {page === 'input' && <InputNote meta={{ ...meta, llm_online: llmOnline }} mode={mode} busy={busy} error={error}
            onAnalyze={(t, l, s) => runAnalysis(t, l, s)} />}
          {page === 'triage' && analysis && <Triage a={analysis} meta={meta} reviewer={reviewer.name} onDecide={decide} />}
          {page === 'diff' && analysis && note && <MedDiff key={analysis.report.meta.report_hash} a={analysis} meta={meta}
            reviewer={reviewer.name} busy={busy} onDecide={decide}
            onPriorList={p => runAnalysis(note.text, `${note.label} + prior list`, note.source, p)} />}
          {page === 'meds' && analysis && note && <Medications a={analysis} analyzedAt={note.at} />}
          {page === 'audit' && <AuditSystem meta={meta} mode={mode} setMode={setMode} llmOnline={llmOnline} refreshKey={auditKey} />}
          {page !== 'input' && page !== 'audit' && !analysis && (
            <div className="mx-auto max-w-md pt-20 text-center text-sm text-muted">Analyze a note first.</div>
          )}
        </main>
      </div>
    </div>
  )
}
