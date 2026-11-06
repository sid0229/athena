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
  analyze: (text: string, mode: string, prior_list?: string) =>
    req<Analysis>('/api/analyze', { text, mode, prior_list }),
  decide: (body: { text: string; mode: string; finding_id: string; reviewer: string; action: string;
    reason_code?: string | null; comment?: string | null }) =>
    req<{ decisions: Record<string, Decision> }>('/api/decision', body),
  audit: () => req<{ chain_ok: boolean; broken_at: number | null; entries: Decision[] }>('/api/audit'),
}
