import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { Sidebar, type Page } from './components/Sidebar'
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
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (document.documentElement.dataset.theme as 'light' | 'dark') || 'light')
  const [reviewer, setReviewerState] = useState<Reviewer>(() => stored('athena-reviewer', { name: '', role: '' }))
  const [mode, setModeState] = useState<string>(() => stored('athena-mode', 'hybrid'))
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [analysisMode, setAnalysisMode] = useState(mode)  // mode the current report was made with
  const [note, setNote] = useState<{ text: string; label: string; source: string; at: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [llmOnline, setLlmOnline] = useState(false)
  const [auditKey, setAuditKey] = useState(0)

  useEffect(() => { api.meta().then(m => { setMeta(m); setLlmOnline(m.llm_online) }).catch(e => setMetaError(String(e.message ?? e))) }, [])
  useEffect(() => {
    const id = setInterval(() => api.status().then(s => setLlmOnline(s.llm_online)).catch(() => {}), 15000)
    return () => clearInterval(id)
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
      setAnalysisMode(mode)
      setNote({ text, label, source, at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })
      if (!prior) setPage('triage')
    } catch (e) {
      setError(`Analysis failed: ${(e as Error).message}`)
    } finally { setBusy(false) }
  }, [mode])

  const decide = useCallback(async (f: Finding, action: 'confirm' | 'override', reason?: string, comment?: string) => {
    if (!analysis) return
    const r = await api.decide({ text: analysis.text, mode: analysisMode, finding_id: f.id, reviewer: reviewer.name, action,
      reason_code: reason ?? null, comment: comment ?? null })
    setAnalysis({ ...analysis, decisions: r.decisions })
    setAuditKey(k => k + 1)
  }, [analysis, analysisMode, reviewer.name])

  if (metaError) return (
    <div className="grid h-full place-items-center p-8 text-center">
      <div><div className="text-[16px] font-semibold text-ink">Cannot reach the local Athena server</div>
        <p className="mt-2 text-[13.5px] text-muted">Start it with <code className="font-mono">python app/server.py</code>, then reload. ({metaError})</p></div>
    </div>
  )
  if (!meta) return <div className="grid h-full place-items-center text-muted"><Spinner size={20} /></div>

  const pending = (types: Set<string>) =>
    analysis ? analysis.report.findings.filter(f => types.has(f.type) && ACTIONABLE(f) && !analysis.decisions[f.id]).length : 0

  return (
    <div className="min-h-full">
      <Sidebar page={page} setPage={setPage} version={meta.version} reviewer={reviewer} setReviewer={setReviewer}
        theme={theme} toggleTheme={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))} hasReport={!!analysis}
        badges={{ triage: pending(TRIAGE_TYPES), diff: pending(RECON_TYPES) }} />
      <div className="pl-[248px]">
        <header className="sticky top-0 z-10 flex h-[68px] items-center justify-between border-b border-line bg-surface/90 px-8 backdrop-blur">
          <div className="meta truncate">
            {note && page !== 'input' && page !== 'audit' ? <>{note.source} · <span className="text-ink-2">{note.label}</span> · analyzed {note.at}</> : null}
          </div>
          <div title={reviewer.name ? `${reviewer.name} · ${reviewer.role}` : 'No reviewer set'}
            className="grid size-10 place-items-center rounded-full bg-accent text-[13px] font-semibold text-accent-ink">
            {reviewer.name ? initials(reviewer.name) : '?'}
          </div>
        </header>
        <main className="px-8 py-10">
          {page === 'input' && <InputNote meta={{ ...meta, llm_online: llmOnline }} mode={mode} busy={busy} error={error}
            onAnalyze={(t, l, s) => runAnalysis(t, l, s)} />}
          {page === 'triage' && analysis && <Triage a={analysis} meta={meta} reviewer={reviewer.name} onDecide={decide} />}
          {page === 'diff' && analysis && note && <MedDiff key={analysis.report.meta.report_hash} a={analysis} meta={meta}
            reviewer={reviewer.name} busy={busy} onDecide={decide}
            onPriorList={p => runAnalysis(note.text, `${note.label} + prior list`, note.source, p)} />}
          {page === 'meds' && analysis && note && <Medications a={analysis} analyzedAt={note.at} />}
          {page === 'audit' && <AuditSystem meta={meta} mode={mode} setMode={setMode} llmOnline={llmOnline} refreshKey={auditKey} />}
          {page !== 'input' && page !== 'audit' && !analysis && (
            <div className="mx-auto max-w-md pt-20 text-center text-[13.5px] text-muted">Analyze a note first.</div>
          )}
        </main>
      </div>
    </div>
  )
}
