import type { Analysis, Finding, Medication } from './types'

export const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s)

/** 'acetylsalicylic acid' -> 'Aspirin (acetylsalicylic acid)'; handles 'a + b'. Display only. */
export function disp(name: string, names: Record<string, string>): string {
  return name.split(' + ').map(p => {
    const part = p.trim()
    const common = names[part.toLowerCase()]
    return common ? `${cap(common)} (${part})` : cap(part)
  }).join(' + ')
}

/** 'Major interaction: warfarin + acetylsalicylic acid' -> display pair. */
export function findingTitle(f: Finding, a: Analysis): string {
  const rest = f.title.split(': ').slice(1).join(': ') || f.title
  if (f.type === 'INTERACTION') return disp(rest, a.display_names)
  const med = a.report.medications.find(m => f.meds.includes(m.id) && rest.toLowerCase().startsWith(m.name.toLowerCase()))
  return med ? disp(med.name, a.display_names) + rest.slice(med.name.length) : cap(rest)
}

export function findingSummary(f: Finding): string {
  const ev = f.evidence
  if (f.type === 'INTERACTION') {
    if (ev.mechanisms?.length) return ev.mechanisms[0].text
    if (ev.ddinter_level) return `DDInter grades this pair ${ev.ddinter_level}.`
  }
  return f.explanation.find(x => !x.startsWith('Risk ')) ?? ''
}

export function minConfidence(f: Finding, meds: Medication[]): number | null {
  const vals = meds.filter(m => f.meds.includes(m.id)).map(m => m.confidence)
  return vals.length ? Math.min(...vals) : null
}

export const ACTIONABLE = (f: Finding) => f.tier === 'critical' || f.tier === 'review'
export const TRIAGE_TYPES = new Set(['INTERACTION', 'DUPLICATE_THERAPY'])
export const RECON_TYPES = new Set(['OMISSION', 'NEW_MEDICATION', 'DOSE_CHANGE'])
export const DECIDABLE = (f: Finding) => f.type !== 'NOT_CHECKED' && f.type !== 'NOT_COVERED'

/** Line of the note that contains a character offset (for "as written" context). */
export function lineAt(text: string, pos: number): string {
  const s = text.lastIndexOf('\n', pos - 1) + 1
  const e = text.indexOf('\n', pos)
  return text.slice(s, e === -1 ? text.length : e).trim()
}

export const fmtTs = (iso: string) => iso.replace('T', ' ').replace('+00:00', '') + ' UTC'

export function initials(name: string): string {
  const skip = new Set(['dr', 'dr.', 'mr', 'mrs', 'ms', 'prof', 'prof.'])
  const w = name.split(/\s+/).filter(x => x && !skip.has(x.toLowerCase()))
  return (w.length ? (w[0][0] + (w.length > 1 ? w[w.length - 1][0] : '')) : '?').toUpperCase()
}

export function shortHash(s: string, n = 6) { return s ? `${s.slice(0, n)}…` : '' }

export async function sha256(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('')
}
