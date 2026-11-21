import type { Analysis, Decision, Meta } from './types'

async function req<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try { msg = (await res.json()).detail ?? msg } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  meta: () => req<Meta>('/api/meta'),
  status: () => req<{ llm_online: boolean }>('/api/status'),
  analyze: (text: string, mode: string, prior_list?: string, input_source?: string, ocr_engine?: string | null) =>
    req<Analysis>('/api/analyze', { text, mode, prior_list, input_source: input_source ?? 'digital', ocr_engine }),
  decide: (body: { report_hash: string; finding_id: string; reviewer: string; action: string;
    reason_code?: string | null; comment?: string | null }) =>
    req<{ decisions: Record<string, Decision> }>('/api/decision', body),
  correct: (body: { report_hash: string; reviewer: string; medications: any[];
    comment?: string | null; text?: string | null; input_source?: string; ocr_engine?: string | null }) =>
    req<Analysis>('/api/correct', body),
  extract: async (file: File, engine?: string) => {
    const q = new URLSearchParams({ filename: file.name })
    if (engine) q.set('ocr_engine', engine)
    const res = await fetch(`/api/extract?${q.toString()}`, { method: 'POST', body: file })
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`
      try { msg = (await res.json()).detail ?? msg } catch { /* not JSON */ }
      throw new Error(msg)
    }
    return res.json() as Promise<{
      name: string
      text: string
      source: 'digital' | 'ocr'
      ocr_engine: string | null
      low_confidence_words: { word: string; confidence: number; start: number; end: number; page: number; bbox?: number[] | null }[]
      page_count: number
      seconds: number
      preview_images: string[]
    }>
  },
  audit: () => req<{ chain_ok: boolean; broken_at: number | null; entries: Decision[] }>('/api/audit'),
}
