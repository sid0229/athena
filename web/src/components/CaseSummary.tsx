import type { Analysis, Finding, Meta } from '../types'
import { ACTIONABLE, DECIDABLE, RECON_TYPES, TRIAGE_TYPES, disp, findingSummary, findingTitle, fmtTs, shortHash } from '../lib'

/** Printed case summary after review (browser print → "Save as PDF"). Always black on white. */
export function CaseSummary({ a, meta, note, reviewer }: {
  a: Analysis; meta: Meta; reviewer: { name: string; role: string }
  note: { label: string; source: string; at: string }
}) {
  const r = a.report
  const byRisk = (x: Finding, y: Finding) => y.risk - x.risk
  const interactions = r.findings.filter(f => TRIAGE_TYPES.has(f.type) && ACTIONABLE(f)).sort(byRisk)
  const recon = r.findings.filter(f => RECON_TYPES.has(f.type) && ACTIONABLE(f)).sort(byRisk)
  const decidable = r.findings.filter(f => DECIDABLE(f) && ACTIONABLE(f))
  const pending = decidable.filter(f => !a.decisions[f.id])
  const overridden = decidable.filter(f => a.decisions[f.id]?.action === 'override').length
  const unchecked = r.findings.filter(f => f.type === 'NOT_CHECKED' || f.type === 'NOT_COVERED')
  const active = r.medications.filter(m => m.status !== 'mentioned')

  return (
    <article className="case-summary mx-auto max-w-[780px] bg-white px-2 text-[11pt] leading-snug text-black">
      <header className="flex items-start justify-between border-b-2 border-black pb-2">
        <div>
          <h1 className="text-[18pt] font-bold">Athena — medication review summary</h1>
          <p className="text-[10pt]">{note.source} · {note.label} · {a.case_id}</p>
        </div>
        <div className="text-right text-[9pt]">
          <div>Printed {new Date().toLocaleString()}</div>
          <div>Analyzed {note.at}</div>
        </div>
      </header>

      <section className="mt-3 grid grid-cols-4 gap-2 text-[10pt]">
        <Box label="Medications" value={`${active.length} active / ${r.medications.length}`} />
        <Box label="Critical / Review" value={`${r.meta.counts.tiers.critical ?? 0} / ${r.meta.counts.tiers.review ?? 0}`} />
        <Box label="Decided" value={`${decidable.length - pending.length} of ${decidable.length}`} />
        <Box label="Overridden" value={String(overridden)} />
      </section>
      {pending.length > 0 && (
        <p className="mt-2 border-2 border-black px-2 py-1 text-[10pt] font-semibold">
          ⚠ Review incomplete: {pending.length} actionable finding{pending.length === 1 ? '' : 's'} not yet confirmed or overridden.
        </p>
      )}

      <Findings title="Interactions" items={interactions} a={a} reasons={meta.override_reasons} />
      <Findings title="Reconciliation (admission → discharge)" items={recon} a={a} reasons={meta.override_reasons} />

      {unchecked.length > 0 && (
        <section className="mt-4 break-inside-avoid">
          <h2 className="border-b border-black text-[12pt] font-bold">Not fully checked</h2>
          {unchecked.map(f => (
            <p key={f.id} className="mt-1 text-[10pt]"><b>{f.title}:</b> {f.explanation.slice(0, -1).join('; ')}</p>
          ))}
        </section>
      )}

      <section className="mt-4">
        <h2 className="border-b border-black text-[12pt] font-bold">Medication list</h2>
        <table className="mt-1 w-full border-collapse text-[9.5pt]">
          <thead><tr className="text-left">
            <th className="border-b border-black py-0.5 pr-2">Medication</th><th className="border-b border-black pr-2">Dose / route / frequency</th>
            <th className="border-b border-black pr-2">Lists</th><th className="border-b border-black pr-2">Status</th>
            <th className="border-b border-black text-right">Confidence</th>
          </tr></thead>
          <tbody>
            {r.medications.map(m => (
              <tr key={m.id} className="break-inside-avoid">
                <td className="border-b border-[#bbb] py-0.5 pr-2 align-top">{disp(m.name, a.display_names)}</td>
                <td className="border-b border-[#bbb] pr-2 align-top">{[m.attributes.Strength, m.attributes.Route, m.attributes.Frequency].filter(Boolean).join(' · ') || '—'}</td>
                <td className="whitespace-nowrap border-b border-[#bbb] pr-2 align-top">{[m.in_admission && 'Adm', m.in_discharge && 'Dis'].filter(Boolean).join(' / ') || '—'}</td>
                <td className="border-b border-[#bbb] pr-2 align-top capitalize">{m.status}</td>
                <td className="border-b border-[#bbb] text-right align-top">{m.confidence.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="mt-5 break-inside-avoid text-[10pt]">
        <h2 className="border-b border-black text-[12pt] font-bold">Sign-off</h2>
        <div className="mt-2 grid grid-cols-2 gap-x-8 gap-y-5">
          <div>Reviewer: <b>{reviewer.name || '________________________'}</b>{reviewer.role ? ` · ${reviewer.role}` : ''}</div>
          <div>Signature: ________________________</div>
          <div>Date / time: ________________________</div>
          <div>Discussed with prescriber: ☐ yes ☐ no</div>
        </div>
      </section>

      <footer className="mt-5 border-t border-black pt-1.5 text-[8pt] leading-tight">
        Engine: Athena {meta.version} · {meta.modes[r.meta.extraction_mode] ?? r.meta.extraction_mode}{r.meta.llm_model ? ` · ${r.meta.llm_model}` : ''} ·
        KB {meta.kb.concepts.toLocaleString()} drugs / {meta.kb.pairs.toLocaleString()} pairs ({meta.kb.sources ?? 'DrugBank, DDInter'}) ·
        report {shortHash(r.meta.report_hash, 16)}. Interaction flags come from DrugBank and DDInter, never from the language model.
        Decisions are recorded in a hash-chained audit log. Decision support only — not a medical device; the reviewing clinician is responsible for all decisions.
      </footer>
    </article>
  )
}

const Box = ({ label, value }: { label: string; value: string }) => (
  <div className="border border-black px-2 py-1"><div className="text-[8pt] uppercase tracking-wide">{label}</div><div className="font-bold">{value}</div></div>
)

function Findings({ title, items, a, reasons }: { title: string; items: Finding[]; a: Analysis; reasons: Record<string, string> }) {
  if (!items.length) return (
    <section className="mt-4"><h2 className="border-b border-black text-[12pt] font-bold">{title}</h2>
      <p className="mt-1 text-[10pt]">No critical or review findings.</p></section>
  )
  return (
    <section className="mt-4">
      <h2 className="border-b border-black text-[12pt] font-bold">{title}</h2>
      {items.map(f => {
        const d = a.decisions[f.id]
        const ev = f.evidence
        return (
          <div key={f.id} className="break-inside-avoid border-b border-[#bbb] py-1.5">
            <div className="flex items-baseline justify-between gap-3">
              <div className="font-semibold">
                <span className="mr-1.5 inline-block min-w-[58px] border border-black px-1 text-center text-[8pt] uppercase">{f.tier}</span>
                {f.type === 'DUPLICATE_THERAPY' ? 'Duplicate therapy: ' : ''}{findingTitle(f, a)}{ev.severity ? ` — ${ev.severity}${ev.severity_basis === 'rule' ? ' (high-alert rule)' : ev.severity_basis === 'derived' ? ' (derived)' : ''}` : ''}
              </div>
              <div className="shrink-0 text-[9pt]">risk {f.risk.toFixed(2)}</div>
            </div>
            <p className="mt-0.5 text-[9.5pt]">{findingSummary(f)}</p>
            {ev.rule && <p className="text-[9pt]">Rule: {ev.rule.name} — {ev.rule.source}</p>}
            {ev.sources?.length ? <p className="text-[9pt]">Sources: {ev.sources.map(s => (s === 'ddinter' ? 'DDInter' : s === 'rule' ? 'High-alert rule' : 'DrugBank')).join(', ')}</p> : null}
            <p className="mt-0.5 text-[9.5pt]">
              {d ? <><b>{d.action === 'confirm' ? '✓ Confirmed' : '✗ Overridden'}</b> by {d.reviewer} · {fmtTs(d.ts)}
                {d.action === 'override' && d.reason_code ? ` · ${reasons[d.reason_code] ?? d.reason_code}` : ''}{d.comment ? ` — “${d.comment}”` : ''}
                <span className="text-[8pt]"> · audit {shortHash(d.row_hash, 10)}</span></>
                : <b>☐ Not yet reviewed</b>}
            </p>
          </div>
        )
      })}
    </section>
  )
}
