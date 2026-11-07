import { Download, ShieldCheck, TriangleAlert } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { Decision, Meta } from '../types'
import { fmtTs, shortHash } from '../lib'
import { Button, Spinner, cx } from '../components/ui'

export function AuditSystem({ meta, mode, setMode, llmOnline, refreshKey }: {
  meta: Meta; mode: string; setMode: (m: string) => void; llmOnline: boolean; refreshKey: number
}) {
  const [log, setLog] = useState<{ chain_ok: boolean; broken_at: number | null; entries: Decision[] } | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  const [verifying, setVerifying] = useState(false)
  const [verified, setVerified] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const load = useCallback(() => api.audit().then(r => { setLog(r); setLoadError(null) })
    .catch(e => setLoadError(`Could not load the audit log: ${(e as Error).message}`)), [])
  useEffect(() => { load(); setVerified(null) }, [load, refreshKey])  // a new decision makes "verified" stale

  const verify = async () => {
    setVerifying(true)
    try {
      const r = await api.audit()
      setLog(r)
      setVerified(r.chain_ok ? `Chain intact · ${r.entries.length} entries verified` : `Chain broken at row ${r.broken_at}`)
    } catch (e) {
      setVerified(`Verification failed: ${(e as Error).message}`)
    } finally { setVerifying(false) }
  }
  // entries are UTC; "today" is the reviewer's local day
  const nToday = log?.entries.filter(e => new Date(e.ts).toDateString() === new Date().toDateString()).length ?? 0

  return (
    <div className="mx-auto max-w-[1320px]">
      <Section title="Audit Log" sub="Decision ledger · append-only, hash-chained"
        right={<>
          <span className="eyebrow">{nToday} entr{nToday === 1 ? 'y' : 'ies'} recorded today</span>
          <a href="/api/audit.csv" download
            className="inline-flex h-10 items-center gap-1.5 rounded-md border border-line-strong bg-surface px-4 text-sm font-medium text-ink hover:bg-surface-2">
            <Download size={15} strokeWidth={1.75} aria-hidden />Export CSV</a>
        </>} />
      <div className="border-y border-line">
        {loadError && <div role="alert" className="px-3 py-6 text-sm text-critical">{loadError}</div>}
        {!log && !loadError && <div className="px-3 py-6"><Spinner /></div>}
        {log && log.entries.length === 0 && <div className="px-3 py-8 text-center text-sm text-muted">No decisions recorded yet.</div>}
        {log?.entries.map(e => {
          const isOpen = open === e.id
          const target = e.title.split(': ').slice(1).join(': ') || e.title
          return (
            <div key={e.id} className="border-b border-line last:border-b-0">
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-3 py-3">
                <span className="meta">{fmtTs(e.ts)}</span>
                <span className="text-sm text-muted">Reviewer <span className="text-ink">{e.reviewer}</span></span>
                <span className="min-w-0 flex-1 truncate text-sm text-muted">Target <span className="text-ink">{target}</span></span>
                <span className={cx('flex items-center gap-1.5 text-sm', e.action === 'confirm' ? 'text-ok' : 'text-critical')}>
                  <span className="size-2 bg-current" />{e.action === 'confirm' ? 'Confirmed' : 'Overridden with clinical reason'}
                </span>
                <button onClick={() => setOpen(isOpen ? null : e.id)} className="meta hover:text-ink cursor-pointer">
                  [{isOpen ? '− hide' : '+ details'}]
                </button>
              </div>
              {isOpen && (
                <div className="mx-3 mb-3 rounded-md border border-line bg-surface-2 px-3.5 py-3">
                  <div className="flex items-center justify-between">
                    <span className="eyebrow">{e.case_id} · {e.finding_type.replace('_', ' ').toLowerCase()}</span>
                    <span className={cx('eyebrow', e.tier === 'critical' ? 'text-critical' : e.tier === 'review' ? 'text-review' : '')}>Tier: {e.tier}</span>
                  </div>
                  <p className="mt-1.5 text-sm text-ink">
                    {e.action === 'override' ? (meta.override_reasons[e.reason_code ?? ''] ?? 'Override') : 'Finding confirmed by the reviewer.'}
                    {e.comment ? ` — ${e.comment}` : ''}
                  </p>
                  <p className="meta mt-2">{e.title} · report {shortHash(e.report_hash, 12)} · row {shortHash(e.row_hash, 12)}</p>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="mt-14" />
      <Section title="System & Extraction Engine" sub="Local runtime configuration" right={<span className="eyebrow">On-device · localhost only</span>} />
      <div className="border-y border-line">
        <Row title="Extraction engine" desc="How medications are read from the note. Interaction flags always come from the databases.">
          <div className="flex rounded-md border border-line bg-surface-2 p-0.5">
            {Object.entries(meta.modes).map(([k, v]) => (
              <button key={k} onClick={() => setMode(k)} aria-pressed={mode === k}
                className={cx('rounded px-3 py-1.5 font-mono text-xs cursor-pointer',
                  mode === k ? 'bg-surface text-ink shadow-sm' : 'text-muted hover:text-ink')}>{v}</button>
            ))}
          </div>
        </Row>
        <Row title="Local runtime status"
          desc={<span className="meta flex items-center gap-1.5"><span className={cx('size-2', llmOnline ? 'bg-ok' : 'bg-critical')} />
            {meta.model} · {llmOnline ? 'Online' : 'Offline — rules fallback'} · Ollama on localhost (Apple Metal)</span>}>
          <span className="meta rounded-md border border-line px-2.5 py-1.5">localhost only</span>
        </Row>
        <Row title="Knowledge base"
          desc={<span className="meta">DrugBank + DDInter · {meta.kb.concepts.toLocaleString()} drugs · {meta.kb.pairs.toLocaleString()} interacting pairs{meta.kb.built ? ` · built ${meta.kb.built}` : ''}</span>}>
          <div className="flex items-center gap-3">
            {verified && <span className={cx('meta', log?.chain_ok ? 'text-ok' : 'text-critical')}>{verified}</span>}
            <Button onClick={verify} disabled={verifying} icon={verifying ? <Spinner /> : <ShieldCheck size={15} strokeWidth={1.75} />}>Verify audit chain</Button>
          </div>
        </Row>
      </div>
      {log && !log.chain_ok && (
        <div className="mt-4 flex items-center gap-2 rounded-md border border-critical-line bg-critical-soft px-3 py-2 text-sm text-critical">
          <TriangleAlert size={15} />The audit log was modified outside Athena (chain broken at row {log.broken_at}).
        </div>
      )}
      <div className="mt-8 flex items-center justify-between border-t border-line pt-4">
        <span className="eyebrow">Ledger head: {log?.entries[0] ? shortHash(log.entries[0].row_hash, 12) : '—'}</span>
        <span className="eyebrow">Decision support only · not a medical device</span>
      </div>
    </div>
  )
}

const Section = ({ title, sub, right }: { title: string; sub: string; right?: React.ReactNode }) => (
  <div className="mb-3 flex items-end justify-between gap-4">
    <h2 className="flex items-baseline gap-3 text-xl font-semibold tracking-tight text-ink">{title}<span className="eyebrow">{sub}</span></h2>
    <div className="flex items-center gap-4">{right}</div>
  </div>
)

const Row = ({ title, desc, children }: { title: string; desc: React.ReactNode; children: React.ReactNode }) => (
  <div className="flex items-center justify-between gap-6 border-b border-line px-3 py-4 last:border-b-0">
    <div><div className="text-base font-medium text-ink">{title}</div><div className="mt-1 text-sm text-muted">{desc}</div></div>
    {children}
  </div>
)

