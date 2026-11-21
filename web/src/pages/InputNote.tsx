import { Activity, ArrowRight, ChevronsUpDown, Clipboard, Upload, User } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Meta } from '../types'
import { sha256, shortHash } from '../lib'
import { api } from '../api'
import { Button, Card, ICON, PageHeader, Spinner, Tabs, cx } from '../components/ui'

type Src = 'demo' | 'paste' | 'upload'

interface UploadData {
  name: string
  text: string
  source: 'digital' | 'ocr'
  ocr_engine?: string | null
  low_confidence_words: { word: string; confidence: number; start: number; end: number; page: number; bbox?: number[] | null }[]
  preview_images: string[]
}

export function InputNote({ meta, mode, busy, error, onAnalyze }: {
  meta: Meta; mode: string; busy: boolean; error: string | null
  onAnalyze: (text: string, label: string, source: string, prior?: string, input_source?: string, ocr_engine?: string | null) => void
}) {
  const [src, setSrc] = useState<Src>('demo')
  const [demo, setDemo] = useState(meta.demos[0]?.id ?? '')
  const [pasted, setPasted] = useState('')
  const [upload, setUpload] = useState<UploadData | null>(null)
  const [hash, setHash] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [reading, setReading] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  async function readFile(f: File) {
    setFileError(null)
    const ext = f.name.toLowerCase().split('.').pop()
    if (ext === 'txt') {
      const text = await f.text()
      setUpload({ name: f.name, text, source: 'digital', low_confidence_words: [], preview_images: [] })
      return
    }
    const validExts = ['docx', 'pdf', 'png', 'jpg', 'jpeg', 'heic']
    if (!validExts.includes(ext || '')) {
      setFileError('Unsupported file type: upload a .txt, .docx, .pdf, .png, .jpg, .jpeg or .heic file.')
      return
    }
    setReading(true)
    try {
      const res = await api.extract(f)
      setUpload(res)
    } catch (e) {
      setFileError((e as Error).message)
    } finally {
      setReading(false)
    }
  }
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
      <PageHeader title="Med Input" right={src === 'demo' && (
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
        { id: 'upload', label: 'Upload File', icon: <Upload size={14} strokeWidth={1.75} /> },
      ]} />

      <Card className="mt-6 overflow-hidden">
        <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
          <div className="eyebrow flex items-center gap-2">
            <span className={cx('size-1.5 rounded-full', text ? 'bg-ok' : 'bg-faint')} />
            Source document · raw plaintext
            {src === 'demo' && <span className="normal-case tracking-normal text-faint">· synthetic, fictional patient</span>}
            {src === 'upload' && upload && <span className="normal-case tracking-normal text-faint">· {upload.name} · editable, check the text before analysis</span>}
          </div>
          <div className="flex items-center gap-4">
            <span className="meta">UTF-8</span>
            {src === 'paste' && pasted && <button onClick={() => setPasted('')} className="meta hover:text-ink cursor-pointer">Clear</button>}
            {src === 'upload' && upload && <button onClick={() => { setUpload(null); setFileError(null) }} className="meta hover:text-ink cursor-pointer">Clear</button>}
          </div>
        </div>

        {src === 'upload' && upload?.source === 'ocr' && (
          <div className="flex items-center justify-between border-b border-accent/25 bg-accent/10 px-4 py-2.5 text-sm text-accent">
            <div className="flex items-center gap-2">
              <span className="font-semibold">Text read by OCR.</span>
              <span>Check it against the original before analysis.</span>
            </div>
            {upload.ocr_engine && (
              <span className="rounded bg-accent/20 px-2 py-0.5 text-xs font-mono font-medium uppercase tracking-wider">
                Engine: {upload.ocr_engine}
              </span>
            )}
          </div>
        )}

        {src === 'demo' && (
          <pre className="scroll-thin h-[400px] overflow-auto whitespace-pre-wrap px-4 py-4 font-sans text-sm leading-[1.75] text-ink-2">{text}</pre>
        )}
        {src === 'paste' && (
          <textarea aria-label="Discharge summary text" value={pasted} onChange={e => setPasted(e.target.value)} placeholder="Paste a discharge summary…"
            className="scroll-thin block h-[400px] w-full resize-y bg-transparent px-4 py-4 text-sm leading-[1.75] text-ink-2 outline-none placeholder:text-faint" />
        )}
        {src === 'upload' && (upload ? (
          <div className={cx(upload.preview_images?.length ? 'grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-line' : '')}>
            {upload.preview_images?.length > 0 && (
              <div className="p-4 bg-surface-2/40 overflow-auto max-h-[460px] flex flex-col items-center justify-start">
                <div className="eyebrow self-start mb-2">Original Document Preview</div>
                <img src={upload.preview_images[0]} alt="Uploaded document" className="rounded border border-line shadow-sm max-h-[400px] object-contain" />
              </div>
            )}
            <div className="flex flex-col">
              <textarea aria-label={`Text read from ${upload.name} — check and correct before analysis`} value={upload.text}
                onChange={e => setUpload({ ...upload, text: e.target.value })}
                className="scroll-thin block h-[380px] w-full resize-y bg-transparent px-4 py-4 text-sm leading-[1.75] text-ink-2 outline-none" />
              {upload.low_confidence_words.length > 0 && (
                <div className="border-t border-line px-4 py-2 bg-surface-2/60 text-xs">
                  <div className="font-medium text-warn mb-1">Low-confidence words to verify ({upload.low_confidence_words.length}):</div>
                  <div className="flex flex-wrap gap-1.5 max-h-[70px] overflow-auto">
                    {upload.low_confidence_words.map((w, i) => (
                      <span key={i} className="rounded bg-warn/15 border border-warn/30 px-1.5 py-0.5 font-mono text-warn">
                        {w.word} ({Math.round(w.confidence * 100)}%)
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <button onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={async e => {
              e.preventDefault()
              const f = e.dataTransfer.files[0]
              if (f) await readFile(f)
            }}
            className="flex h-[400px] w-full flex-col items-center justify-center gap-2 text-muted hover:text-ink cursor-pointer">
            {reading ? <Spinner /> : <Upload size={22} strokeWidth={1.5} />}
            <span className="text-sm">{reading ? 'Reading file…' : 'Drop a note, scanned PDF, or photo (.png, .jpg, .heic) or click to choose'}</span>
            <span className="meta">Read locally on this device · Apple Vision OCR for scans & photos</span>
            {fileError && <span role="alert" className="mt-2 rounded-md border border-critical-line bg-critical-soft px-3 py-1.5 text-sm text-critical">{fileError}</span>}
          </button>
        ))}
        <input ref={fileRef} type="file" hidden
          accept=".txt,.docx,.pdf,.png,.jpg,.jpeg,.heic,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/png,image/jpeg,image/heic"
          onChange={async e => {
            const f = e.target.files?.[0]
            if (f) await readFile(f)
            e.target.value = ''  // picking the same file again after Clear must work
          }} />

        <div className="flex items-center justify-between gap-4 border-t border-line px-4 py-3">
          <div className="meta flex flex-wrap items-center gap-x-2">
            <span>{stats}</span><span className="text-faint">·</span><span>Local analysis</span>
            <span className="text-faint">·</span><span className="text-faint">no network calls</span>
          </div>
          <Button kind="primary" className="h-11 px-5" disabled={!text.trim() || busy}
            onClick={() => onAnalyze(text, label, source, undefined, upload?.source, upload?.ocr_engine)}
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
        {hash && <span className="meta">Note SHA-256: <span className="font-mono text-ink-2">{shortHash(hash)}</span></span>}
      </div>
    </div>
  )
}
