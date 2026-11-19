import { Check, ChevronDown, ChevronUp, FileText, Pencil, Plus, Trash2, X } from 'lucide-react'
import { Fragment, useEffect, useRef, useState } from 'react'
import type { Analysis, Medication } from '../types'
import { cap, disp } from '../lib'
import { Button, Card, PageHeader, Spinner, cx } from '../components/ui'

const STATUS: Record<string, string> = { active: 'text-ok', held: 'text-hold', stopped: 'text-critical', mentioned: 'text-muted' }
const CHECKED = new Set(['mapped'])

interface EditableMed {
  id: string
  name: string
  raw: string
  strength: string
  route: string
  frequency: string
  status: 'active' | 'held' | 'stopped' | 'mentioned'
  in_admission: boolean
  in_discharge: boolean
  isNew?: boolean
}

function toEditable(m: Medication): EditableMed {
  return {
    id: m.id,
    name: m.name,
    raw: m.raw || m.name,
    strength: m.attributes.Strength || '',
    route: m.attributes.Route || '',
    frequency: m.attributes.Frequency || '',
    status: m.status,
    in_admission: m.in_admission,
    in_discharge: m.in_discharge,
  }
}

export function Medications({
  a,
  analyzedAt,
  reviewer = '',
  onCorrect,
}: {
  a: Analysis
  analyzedAt: string
  reviewer?: string
  onCorrect?: (meds: any[], comment?: string) => Promise<void>
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [note, setNote] = useState(false)
  const [editing, setEditing] = useState(false)
  const [items, setItems] = useState<EditableMed[]>(() => a.report.medications.map(toEditable))
  const [comment, setComment] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  // Sync items when new analysis arrives
  useEffect(() => {
    setItems(a.report.medications.map(toEditable))
    setEditing(false)
    setSuccess(false)
    setError(null)
  }, [a.report.meta.report_hash])

  const meds = a.report.medications

  const handleStartEdit = () => {
    setItems(meds.map(toEditable))
    setComment('')
    setError(null)
    setSuccess(false)
    setEditing(true)
  }

  const handleCancel = () => {
    setItems(meds.map(toEditable))
    setError(null)
    setEditing(false)
  }

  const handleAddMed = () => {
    setItems(cur => [
      ...cur,
      {
        id: `new_${Date.now()}`,
        name: '',
        raw: '',
        strength: '',
        route: '',
        frequency: '',
        status: 'active',
        in_admission: false,
        in_discharge: true,
        isNew: true,
      },
    ])
  }

  const handleRemoveMed = (id: string) => {
    setItems(cur => cur.filter(x => x.id !== id))
  }

  const handleFieldChange = (id: string, field: keyof EditableMed, val: any) => {
    setItems(cur =>
      cur.map(x => {
        if (x.id !== id) return x
        const updated = { ...x, [field]: val }
        if (field === 'name' && x.isNew) {
          updated.raw = val
        }
        return updated
      })
    )
  }

  const handleSave = async () => {
    if (!onCorrect) return
    if (!reviewer.trim()) {
      setError('Please set your reviewer name in the sidebar before saving corrections.')
      return
    }
    const emptyDrug = items.find(x => !x.name.trim())
    if (emptyDrug) {
      setError('All medications must have a drug name.')
      return
    }

    setSaving(true)
    setError(null)
    try {
      const payload = items.map(m => ({
        id: m.isNew ? undefined : m.id,
        name: m.name.trim(),
        raw: m.raw.trim() || m.name.trim(),
        status: m.status,
        attributes: {
          ...(m.strength.trim() ? { Strength: m.strength.trim() } : {}),
          ...(m.route.trim() ? { Route: m.route.trim() } : {}),
          ...(m.frequency.trim() ? { Frequency: m.frequency.trim() } : {}),
        },
        in_admission: m.in_admission,
        in_discharge: m.in_discharge,
      }))
      await onCorrect(payload, comment.trim() || undefined)
      setSuccess(true)
      setEditing(false)
    } catch (e) {
      setError(`Failed to save correction: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-[1320px]">
      <PageHeader
        title="Medication Record"
        tag={<span className="rounded border border-line px-1.5 py-px">{meds.length} entries</span>}
        right={
          <div className="flex items-center gap-2">
            {!editing && onCorrect && (
              <Button icon={<Pencil size={15} strokeWidth={1.75} />} onClick={handleStartEdit}>
                Correct Medications
              </Button>
            )}
            <Button icon={<FileText size={15} strokeWidth={1.75} />} onClick={() => setNote(true)}>
              View Source Note
            </Button>
          </div>
        }
      />

      {success && (
        <div className="mb-4 flex items-center justify-between rounded-md border border-ok-line bg-ok-soft px-4 py-3 text-sm text-ok">
          <span className="flex items-center gap-2">
            <Check size={16} /> Corrections saved and interaction checks re-run.
          </span>
          <button onClick={() => setSuccess(false)} className="text-muted hover:text-ink cursor-pointer">
            <X size={15} />
          </button>
        </div>
      )}

      {error && (
        <div className="mb-4 flex items-center justify-between rounded-md border border-critical-line bg-critical-soft px-4 py-3 text-sm text-critical">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-muted hover:text-ink cursor-pointer">
            <X size={15} />
          </button>
        </div>
      )}

      {editing ? (
        <Card className="overflow-hidden p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-4 border-b border-line pb-4">
            <div>
              <div className="text-base font-semibold text-ink">Reviewer Correction Mode</div>
              <div className="meta text-xs">
                Add missed drugs, remove wrongly identified drugs, or edit attributes. Saving re-runs interaction checks without re-extracting.
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button kind="outline" onClick={handleAddMed} icon={<Plus size={15} />}>
                Add Missed Drug
              </Button>
              <Button kind="ghost" onClick={handleCancel} disabled={saving}>
                Cancel
              </Button>
              <Button kind="primary" onClick={handleSave} disabled={saving} icon={saving ? <Spinner size={15} /> : <Check size={15} />}>
                {saving ? 'Saving...' : 'Save & Re-run Checks'}
              </Button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-line text-xs font-medium uppercase tracking-wider text-muted">
                  <th className="py-2.5 pr-3">Drug Name</th>
                  <th className="py-2.5 pr-3">Strength</th>
                  <th className="py-2.5 pr-3">Route</th>
                  <th className="py-2.5 pr-3">Frequency</th>
                  <th className="py-2.5 pr-3">Status</th>
                  <th className="py-2.5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {items.map(m => (
                  <tr key={m.id} className="hover:bg-surface-2">
                    <td className="py-2.5 pr-3">
                      <input
                        type="text"
                        value={m.name}
                        onChange={e => handleFieldChange(m.id, 'name', e.target.value)}
                        placeholder="e.g. Warfarin"
                        className="w-full rounded border border-line bg-surface px-2.5 py-1 text-sm text-ink outline-none focus:border-ink"
                      />
                    </td>
                    <td className="py-2.5 pr-3">
                      <input
                        type="text"
                        value={m.strength}
                        onChange={e => handleFieldChange(m.id, 'strength', e.target.value)}
                        placeholder="e.g. 5 mg"
                        className="w-28 rounded border border-line bg-surface px-2.5 py-1 text-sm text-ink outline-none focus:border-ink"
                      />
                    </td>
                    <td className="py-2.5 pr-3">
                      <input
                        type="text"
                        value={m.route}
                        onChange={e => handleFieldChange(m.id, 'route', e.target.value)}
                        placeholder="e.g. PO"
                        className="w-24 rounded border border-line bg-surface px-2.5 py-1 text-sm text-ink outline-none focus:border-ink"
                      />
                    </td>
                    <td className="py-2.5 pr-3">
                      <input
                        type="text"
                        value={m.frequency}
                        onChange={e => handleFieldChange(m.id, 'frequency', e.target.value)}
                        placeholder="e.g. daily"
                        className="w-28 rounded border border-line bg-surface px-2.5 py-1 text-sm text-ink outline-none focus:border-ink"
                      />
                    </td>
                    <td className="py-2.5 pr-3">
                      <select
                        value={m.status}
                        onChange={e => handleFieldChange(m.id, 'status', e.target.value)}
                        className="rounded border border-line bg-surface px-2 py-1 text-sm text-ink outline-none focus:border-ink cursor-pointer"
                      >
                        <option value="active">Active</option>
                        <option value="held">Held</option>
                        <option value="stopped">Stopped</option>
                        <option value="mentioned">Mentioned</option>
                      </select>
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => handleRemoveMed(m.id)}
                        className="inline-flex items-center gap-1 text-xs text-critical hover:underline cursor-pointer"
                        title="Remove drug"
                      >
                        <Trash2 size={14} /> Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-4 border-t border-line pt-4">
            <div className="flex-1 max-w-md">
              <label htmlFor="correction-comment" className="eyebrow block mb-1">
                Reason / Comment (optional)
              </label>
              <input
                id="correction-comment"
                type="text"
                value={comment}
                onChange={e => setComment(e.target.value)}
                placeholder="e.g. Added missed clarithromycin; corrected warfarin strength"
                className="w-full rounded border border-line bg-surface px-3 py-1.5 text-sm text-ink outline-none focus:border-ink"
              />
            </div>
            <div className="flex items-center gap-2">
              <Button kind="outline" onClick={handleAddMed} icon={<Plus size={15} />}>
                Add Missed Drug
              </Button>
              <Button kind="primary" onClick={handleSave} disabled={saving} icon={saving ? <Spinner size={15} /> : <Check size={15} />}>
                {saving ? 'Saving...' : 'Save & Re-run Checks'}
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-line">
                {['Medication', 'Status', 'Stage', ''].map((h, i) => (
                  <th key={i} className={cx('eyebrow px-4 py-3 font-normal', i === 3 && 'text-right')}>
                    {h || 'Details'}
                  </th>
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
                    <tr
                      onClick={() => setOpen(isOpen ? null : m.id)}
                      className={cx(
                        'cursor-pointer border-b border-line last:border-b-0 hover:bg-surface-2',
                        isOpen && unchecked && 'bg-critical-soft'
                      )}
                    >
                      <td className="px-4 py-3.5">
                        <span className="text-base text-ink">{disp(m.name, a.display_names)}</span>
                        {m.attributes.Strength && <span className="meta ml-2">{m.attributes.Strength}</span>}
                        {unchecked && (
                          <span className="ml-2 inline-flex items-center gap-1 align-middle text-xs text-critical">
                            <span aria-hidden className="inline-block size-1.5 rounded-full bg-critical" />
                            not checked
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3.5">
                        <span className={cx('flex items-center gap-1.5 text-sm', STATUS[m.status])}>
                          <span className="size-1.5 rounded-full bg-current" />
                          <span className="text-ink">{cap(m.status)}</span>
                        </span>
                      </td>
                      <td className="meta px-4 py-3.5">{stage}</td>
                      <td className="px-4 py-3.5 text-right text-muted">
                        <button
                          aria-expanded={isOpen}
                          aria-label={`Details for ${disp(m.name, a.display_names)}`}
                          className="cursor-pointer"
                        >
                          {isOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                        </button>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr className="border-b border-line">
                        <td colSpan={4} className="px-4 pb-4 pt-1">
                          <div className={cx('rounded-md border p-4', unchecked ? 'border-critical-line' : 'border-line')}>
                            <div className="eyebrow mb-1.5">Raw extracted string</div>
                            <div className="meta rounded border border-line bg-surface-2 px-3 py-2 text-ink-2">
                              as written: “{m.raw}”{dose ? ` · ${dose}` : ''}{m.text_status ? ` · note says “${m.text_status}”` : ''}
                            </div>
                            <div className="mt-4 grid grid-cols-2 gap-4">
                              <div>
                                <div className="eyebrow mb-1.5">Mapping status</div>
                                {unchecked ? (
                                  <div className="flex items-center gap-1.5 text-sm text-critical">
                                    <span className="size-1.5 rounded-full bg-critical" />
                                    {m.norm_status === 'drug_class'
                                      ? 'Drug class — interactions not checked'
                                      : m.norm_status === 'ambiguous'
                                      ? `Ambiguous (${m.kb_names.join(', ')}) — not checked`
                                      : 'Not mapped to a known drug — interactions not checked'}
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-1.5 text-sm text-ink">
                                    <span className="size-1.5 rounded-full bg-ok" />
                                    Mapped to {m.kb_names.join(' + ')}{' '}
                                    <span className="meta">({m.norm_stage.replace('_', ' ')} match)</span>
                                  </div>
                                )}
                              </div>
                              <div>
                                <div className="eyebrow mb-1.5">Confidence score</div>
                                <div className="text-sm">
                                  <span className="tabular-nums font-semibold text-ink">{m.confidence.toFixed(2)}</span>
                                  <span className="meta">
                                    {' '}
                                    / 1.00 · {Object.entries(m.confidence_parts).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(' · ')}
                                  </span>
                                </div>
                              </div>
                            </div>
                            {unchecked && (
                              <div className="meta mt-4 border-t border-line pt-3">
                                Action: review this medication manually — Athena could not check its interactions.
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </Card>
      )}

      <div className="mt-5 flex items-center justify-between">
        <span className="meta flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-ok" />
            Verified concept
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-critical" />
            Manual review needed
          </span>
        </span>
        <span className="meta">
          Analyzed {analyzedAt} · {a.report.meta.extraction_mode} · {a.report.meta.extraction_seconds}s
        </span>
      </div>
      {note && <NoteDrawer a={a} onClose={() => setNote(false)} />}
    </div>
  )
}

function NoteDrawer({ a, onClose }: { a: Analysis; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    d?.showModal()
    return () => d?.close()
  }, [])
  const hot = new Set(a.report.findings.filter(f => f.tier === 'critical').flatMap(f => f.meds))
  const spans = a.report.medications
    .flatMap(m => m.mentions.map(x => ({ ...x, id: m.id, name: m.name })))
    .sort((x, y) => x.start - y.start)
  const parts: React.ReactNode[] = []
  let pos = 0
  spans.forEach((s, i) => {
    if (s.start < pos) return
    parts.push(a.text.slice(pos, s.start))
    parts.push(
      <mark key={i} className={cx('drug', hot.has(s.id) && 'hot')} title={disp(s.name, a.display_names)}>
        {a.text.slice(s.start, s.end)}
      </mark>
    )
    pos = s.end
  })
  parts.push(a.text.slice(pos))
  return (
    <dialog
      ref={ref}
      aria-label="Source note"
      onClose={onClose}
      onClick={e => {
        if (e.target === ref.current) onClose()
      }}
      className="fixed inset-y-0 right-0 left-auto m-0 h-full max-h-none w-full max-w-[680px] border-l border-line bg-surface p-0 text-ink backdrop:bg-black/30"
    >
      <div className="flex h-full flex-col">
        <div className="flex h-[68px] items-center justify-between border-b border-line px-5">
          <div>
            <div className="text-base font-semibold text-ink">Source note</div>
            <div className="meta mt-0.5 flex gap-3">
              <span>
                <mark className="drug">drug</mark> extracted
              </span>
              <span>
                <mark className="drug hot">drug</mark> in a critical finding
              </span>
            </div>
          </div>
          <button onClick={onClose} aria-label="Close source note" className="grid size-10 place-items-center rounded-md text-muted hover:text-ink cursor-pointer">
            <X size={18} />
          </button>
        </div>
        <pre className="scroll-thin flex-1 overflow-auto whitespace-pre-wrap px-5 py-4 font-sans text-sm leading-[1.8] text-ink-2">{parts}</pre>
      </div>
    </dialog>
  )
}
