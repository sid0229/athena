export type Tier = 'critical' | 'review' | 'info'

export interface MentionRef { start: number; end: number; text: string; section: string | null; sources: string[] }

export interface Medication {
  id: string
  name: string
  raw: string
  norm_status: string
  norm_stage: string
  kb_names: string[]
  mentions: MentionRef[]
  attributes: Record<string, string>
  in_admission: boolean
  in_discharge: boolean
  text_status: string
  status: 'active' | 'held' | 'stopped' | 'mentioned'
  confidence: number
  confidence_parts: Record<string, number>
  admission_attributes: Record<string, string>
}

export interface Mechanism { text: string; derived_severity: string; pct_major: number; n_overlap_graded: number }

export interface Finding {
  id: string
  type: string
  tier: Tier
  risk: number
  title: string
  meds: string[]
  explanation: string[]
  evidence: {
    severity?: string
    severity_basis?: string
    sources?: string[]
    ddinter_level?: string | null
    mechanisms?: Mechanism[]
    confidence?: Record<string, Record<string, number>>
    admission?: string
    discharge?: string
  }
  needs_verification: boolean
}

export interface Report {
  medications: Medication[]
  findings: Finding[]
  meta: {
    extraction_mode: string
    extraction_seconds: number
    warnings: string[]
    sections: { kind: string; title: string; start: number; end: number }[]
    counts: {
      medications: number; active: number; pairs_checked: number; interactions: number
      no_known_interaction: number; not_covered: number; tiers: Record<Tier, number>
    }
    llm_model: string | null
    report_hash: string
  }
}

export interface Decision {
  id: number; ts: string; reviewer: string; case_id: string; report_hash: string; finding_id: string
  finding_type: string; tier: Tier; title: string; action: 'confirm' | 'override'
  reason_code: string | null; comment: string | null; prev_hash: string; row_hash: string
}

export interface Analysis {
  report: Report
  text: string
  case_id: string
  display_names: Record<string, string>
  decisions: Record<string, Decision>
}

export interface Meta {
  version: string
  model: string
  llm_online: boolean
  kb: { concepts: number; pairs: number; built: string | null; sources: string | null }
  modes: Record<string, string>
  override_reasons: Record<string, string>
  demos: { id: string; name: string; text: string }[]
}

export interface Reviewer { name: string; role: string }
