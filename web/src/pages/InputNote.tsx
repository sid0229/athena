import { Activity, ArrowRight, ChevronsUpDown, Clipboard, Upload, User } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Meta } from '../types'
import { sha256, shortHash } from '../lib'
import { Button, Card, ICON, PageHeader, Spinner, Tabs, cx } from '../components/ui'

type Src = 'demo' | 'paste' | 'upload'

export function InputNote({ meta, mode, busy, error, onAnalyze }: {
  meta: Meta; mode: string; busy: boolean; error: string | null
  onAnalyze: (text: string, label: string, source: string) => void
}) {
  const [src, setSrc] = useState<Src>('demo')
  const [demo, setDemo] = useState(meta.demos[0]?.id ?? '')
  const [pasted, setPasted] = useState('')
  const [upload, setUpload] = useState<{ name: string; text: string } | null>(null)
  const [hash, setHash] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const fileRef = useRef<HTMLInputElement>(null)
  const d = meta.demos.find(x => x.id === demo)

  const text = src === 'demo' ? d?.text ?? '' : src === 'paste' ? pasted : upload?.text ?? ''
  const label = src === 'demo' ? d?.name ?? 'Demo patient' : src === 'paste' ? 'Pasted note' : upload?.name ?? 'Uploaded note'
  const source = { demo: 'Synthetic demo patient', paste: 'Pasted note', upload: 'Uploaded file' }[src]
  const stats = useMemo(() => text ? `${text.split('\n').length} lines · ${text.length.toLocaleString()} chars` : 'No note yet', [text])

  useEffect(() => { let live = true; if (text) sha256(text).then(h => live && setHash(h)); else setHash(''); return () => { live = false } }, [text])
  useEffect(() => {
    if (!busy) { setElapsed(0); return }
    const t0 = Date.now(), id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [busy])

  return (
    <div className="mx-auto max-w-[1320px]">
      <PageHeader title="Input Clinical Note" right={src === 'demo' && (
        <label className="relative flex h-11 items-center gap-2 rounded-md border border-line-strong bg-surface pl-3 pr-8 text-sm">
          <span className="text-muted">Patient:</span>
          <select value={demo} onChange={e => setDemo(e.target.value)} aria-label="Demo patient"
            className="appearance-none bg-transparent font-medium text-ink outline-none cursor-pointer">
            {meta.demos.map(x => <option key={x.id} value={x.id}>{x.name}</option>)}
          </select>
          <ChevronsUpDown size={14} className="pointer-events-none absolute right-2.5 text-muted" />
        </label>
      )} />
      <Tabs<Src> value={src} onChange={setSrc} items={[
        { id: 'demo', label: 'Demo Patient', icon: <User size={14} strokeWidth={1.75} /> },
        { id: 'paste', label: 'Paste Note', icon: <Clipboard size={14} strokeWidth={1.75} /> },
        { id: 'upload', label: 'Upload .txt', icon: <Upload size={14} strokeWidth={1.75} /> },
      ]} />

      <Card className="mt-6 overflow-hidden">
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
          <div className="eyebrow flex items-center gap-2">
            <span className={cx('size-1.5 rounded-full', text ? 'bg-ok' : 'bg-faint')} />
            Source document · raw plaintext
            {src === 'demo' && <span className="normal-case tracking-normal text-faint">· synthetic, fictional patient</span>}
          </div>
          <div className="flex items-center gap-4">
            <span className="meta">UTF-8</span>
            {src === 'paste' && pasted && <button onClick={() => setPasted('')} className="meta hover:text-ink cursor-pointer">Clear</button>}
            {src === 'upload' && upload && <button onClick={() => setUpload(null)} className="meta hover:text-ink cursor-pointer">Clear</button>}
          </div>
        </div>

        {src === 'demo' && (
          <pre className="scroll-thin h-[400px] overflow-auto whitespace-pre-wrap px-4 py-4 font-sans text-sm leading-[1.75] text-ink-2">{text}</pre>
        )}
        {src === 'paste' && (
          <textarea aria-label="Discharge summary text" value={pasted} onChange={e => setPasted(e.target.value)} placeholder="Paste a discharge summary…"
            className="scroll-thin block h-[400px] w-full resize-y bg-transparent px-4 py-4 text-sm leading-[1.75] text-ink-2 outline-none placeholder:text-faint" />
        )}
        {src === 'upload' && (upload ? (
          <pre className="scroll-thin h-[400px] overflow-auto whitespace-pre-wrap px-4 py-4 font-sans text-sm leading-[1.75] text-ink-2">{upload.text}</pre>
        ) : (
          <button onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={async e => {
              e.preventDefault()
              const f = e.dataTransfer.files[0]
              if (f && (f.type === 'text/plain' || f.name.toLowerCase().endsWith('.txt'))) setUpload({ name: f.name, text: await f.text() })
            }}
            className="flex h-[400px] w-full flex-col items-center justify-center gap-2 text-muted hover:text-ink cursor-pointer">
            <Upload size={22} strokeWidth={1.5} />
            <span className="text-sm">Drop a .txt file or click to choose</span>
            <span className="meta">Read in the browser · stays on this device</span>
          </button>
        ))}
        <input ref={fileRef} type="file" accept=".txt,text/plain" hidden
          onChange={async e => {
            const f = e.target.files?.[0]
            if (f) setUpload({ name: f.name, text: await f.text() })
            e.target.value = ''  // picking the same file again after Clear must work
          }} />

        <div className="flex items-center justify-between gap-4 border-t border-line px-4 py-3">
          <div className="meta flex flex-wrap items-center gap-x-2">
            <span>{stats}</span><span className="text-faint">·</span><span>Local analysis</span>
            <span className="text-faint">·</span><span className="text-faint">no network calls</span>
          </div>
          <Button kind="primary" className="h-11 px-5" disabled={!text.trim() || busy}
            onClick={() => onAnalyze(text, label, source)}
            icon={busy ? <Spinner /> : <Activity {...ICON} />}>
            {busy ? `Analyzing… ${elapsed}s` : <>Analyze Note <ArrowRight size={15} strokeWidth={1.75} /></>}
          </Button>
        </div>
      </Card>

      {busy && (
        <p className="mt-3 text-sm text-muted">
          Reading the note on this device and checking every medication pair against DrugBank and DDInter.
          {mode !== 'rules' && ' With the local LLM this takes about a minute.'}
        </p>
      )}
      {error && <p role="alert" className="mt-3 rounded-md border border-critical-line bg-critical-soft px-3 py-2 text-sm text-critical">{error}</p>}
      {!meta.llm_online && mode !== 'rules' && (
        <p className="mt-3 rounded-md border border-hold-line bg-hold-soft px-3 py-2 text-sm text-hold">
          Local LLM is offline — Athena will fall back to rules-only extraction. Start Ollama for hybrid mode.
        </p>
      )}

      <div className="mt-4 flex items-center justify-between px-1">
        <span className="meta">Engine: <span className="text-ink-2">Athena {meta.version} · {meta.modes[mode]}{mode !== 'rules' ? ` · ${meta.model}` : ''}</span></span>
        {hash && <span className="meta">Note SHA-256: <span className="text-ink-2">{shortHash(hash)}</span></span>}
      </div>
    </div>
  )
}
