"""Athena visual system: tokens, local fonts, icons and small HTML helpers.

Design (ui-ux-pro-max: clinical decision support, Swiss minimal, dense dashboard):
calm slate canvas, one teal accent, semantic tier colours that are always paired
with an icon and a text label (never colour alone), Figtree for headings,
Atkinson Hyperlegible for body text. Fonts are served from app/static, so the
dashboard makes no network requests.
"""

from __future__ import annotations

import base64
import html

# Lucide icons (ISC licence), inlined as SVG paths.
_ICON_PATHS = {
    "octagon-alert": '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c0-1.1.9-2 2-2h2"/><path d="M4 8c0-1.1.9-2 2-2h2"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "arrow-right-left": '<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>',
    "circle-help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
}


def _svg_uri(name: str, stroke: float) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def icon(name: str, size: int = 16, stroke: float = 2, cls: str = "") -> str:
    """Icon drawn with CSS mask so it takes the current text colour (inline <svg> is
    stripped by Streamlit's HTML sanitiser)."""
    uri = _svg_uri(name, stroke)
    return (f'<span class="ic {cls}" aria-hidden="true" style="width:{size}px;height:{size}px;'
            f'-webkit-mask:url({uri}) center/contain no-repeat;mask:url({uri}) center/contain no-repeat"></span>')


TIERS = {
    "critical": {"label": "Critical", "icon": "octagon-alert"},
    "review": {"label": "Review", "icon": "triangle-alert"},
    "info": {"label": "Info", "icon": "info"},
}
TYPE_LABELS = {
    "INTERACTION": ("Interaction", "arrow-right-left"),
    "DUPLICATE_THERAPY": ("Duplicate therapy", "copy"),
    "OMISSION": ("Omission", "circle-help"),
    "NEW_MEDICATION": ("New medication", "plus"),
    "DOSE_CHANGE": ("Dose change", "repeat"),
    "NOT_CHECKED": ("Not checked", "circle-help"),
    "NOT_COVERED": ("Not covered", "circle-help"),
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def tier_badge(tier: str) -> str:
    t = TIERS[tier]
    return f'<span class="badge tier-{tier}">{icon(t["icon"], 14)}<span>{t["label"]}</span></span>'


def chip(text: str, kind: str = "neutral", ic: str | None = None) -> str:
    return f'<span class="chip chip-{kind}">{icon(ic, 12) if ic else ""}{esc(text)}</span>'


def meter(value: float, label: str | None = None) -> str:
    pct = max(0, min(100, round(value * 100)))
    lvl = "hi" if value >= 0.85 else "mid" if value >= 0.6 else "lo"
    return (f'<span class="meter" role="img" aria-label="{esc(label or "confidence")} {pct}%">'
            f'<span class="meter-track"><span class="meter-fill m-{lvl}" style="width:{pct}%"></span></span>'
            f'<span class="meter-val">{value:.2f}</span></span>')


def css() -> str:
    return """
<style>
@font-face{font-family:'Figtree';src:url('app/static/fonts/Figtree-var.woff2') format('woff2');font-weight:300 900;font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('app/static/fonts/AtkinsonHyperlegible-400.woff2') format('woff2');font-weight:400;font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('app/static/fonts/AtkinsonHyperlegible-700.woff2') format('woff2');font-weight:700;font-display:swap}

:root{
  --bg:#F5F7FA; --surface:#FFFFFF; --surface-2:#F8FAFC; --ink:#0F172A; --ink-2:#334155; --muted:#64748B;
  --line:#E2E8F0; --line-2:#CBD5E1;
  --accent:#0E7490; --accent-ink:#FFFFFF; --accent-soft:#ECFEFF; --accent-line:#A5F3FC;
  --crit:#B91C1C; --crit-bg:#FEF2F2; --crit-line:#FECACA;
  --rev:#B45309;  --rev-bg:#FFFBEB;  --rev-line:#FDE68A;
  --inf:#334155;  --inf-bg:#F1F5F9;  --inf-line:#CBD5E1;
  --ok:#047857;   --ok-bg:#ECFDF5;   --ok-line:#A7F3D0;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.06);
  --radius:12px; --radius-sm:8px;
  --font-head:'Figtree',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --font-body:'Atkinson Hyperlegible',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --font-mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stMarkdown,[data-testid="stMarkdownContainer"]{font-family:var(--font-body)}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important}
.stApp{background:var(--bg)}
h1,h2,h3,h4,.ath-h{font-family:var(--font-head)!important;color:var(--ink);letter-spacing:-.01em}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1480px}
header[data-testid="stHeader"]{background:transparent}
.ic{display:inline-block;vertical-align:-3px;flex:none;background:currentColor}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}

/* sidebar */
section[data-testid="stSidebar"]{background:#0B1220;border-right:1px solid #111827}
section[data-testid="stSidebar"] *{color:#CBD5E1}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#F8FAFC!important}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#111A2E!important;border-color:#1E293B!important;color:#E2E8F0!important}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{background:#12203A}
[data-testid="stSidebarHeader"]{padding-bottom:0}
[data-testid="stLogo"]{height:40px!important;max-width:100%}
.brand{display:flex;align-items:center;gap:10px;margin:2px 0 14px}
.brand-mark{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#0E7490,#155E75);display:grid;place-items:center;color:#fff;box-shadow:inset 0 0 0 1px rgba(255,255,255,.12)}
.brand-name{font-family:var(--font-head);font-weight:700;font-size:1.25rem;color:#F8FAFC;letter-spacing:-.02em;line-height:1}
.brand-sub{font-size:.72rem;color:#94A3B8;letter-spacing:.06em;text-transform:uppercase;margin-top:3px}
.side-label{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:#64748B!important;margin:14px 0 6px;font-weight:700}
.status-row{display:flex;align-items:center;gap:8px;font-size:.86rem;padding:6px 0}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot-ok{background:#10B981;box-shadow:0 0 0 3px rgba(16,185,129,.18)}
.dot-off{background:#F59E0B;box-shadow:0 0 0 3px rgba(245,158,11,.18)}
.side-foot{font-size:.74rem;color:#64748B!important;line-height:1.5;margin-top:18px;border-top:1px solid #1E293B;padding-top:12px}

/* header */
.page-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:16px}
.eyebrow{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:700}
.page-title{font-family:var(--font-head);font-size:1.75rem;font-weight:700;color:var(--ink);margin:2px 0 4px;letter-spacing:-.02em}
.page-sub{color:var(--muted);font-size:.92rem}
.hash{font-family:var(--font-mono);font-size:.78rem;color:var(--muted);background:var(--surface);border:1px solid var(--line);padding:6px 10px;border-radius:999px;display:inline-flex;gap:6px;align-items:center;white-space:nowrap}

/* KPI tiles */
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:4px 0 18px}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--line-2)}
.kpi.k-critical::before{background:var(--crit)}.kpi.k-review::before{background:var(--rev)}.kpi.k-info::before{background:var(--inf)}.kpi.k-accent::before{background:var(--accent)}
.kpi-label{display:flex;align-items:center;gap:6px;font-size:.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.kpi-value{font-family:var(--font-head);font-size:1.9rem;font-weight:700;color:var(--ink);line-height:1.1;margin-top:6px;font-variant-numeric:tabular-nums}
.kpi-foot{font-size:.8rem;color:var(--muted);margin-top:2px}
.k-critical .kpi-label{color:var(--crit)}.k-review .kpi-label{color:var(--rev)}

/* badges & chips */
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:700;font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;padding:4px 9px;border-radius:999px;border:1px solid}
.tier-critical{color:var(--crit);background:var(--crit-bg);border-color:var(--crit-line)}
.tier-review{color:var(--rev);background:var(--rev-bg);border-color:var(--rev-line)}
.tier-info{color:var(--inf);background:var(--inf-bg);border-color:var(--inf-line)}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:.76rem;padding:3px 8px;border-radius:6px;border:1px solid var(--line);background:var(--surface-2);color:var(--ink-2);margin:0 6px 6px 0;white-space:nowrap}
.chip-accent{background:var(--accent-soft);border-color:var(--accent-line);color:var(--accent)}
.chip-ok{background:var(--ok-bg);border-color:var(--ok-line);color:var(--ok)}
.chip-warn{background:var(--rev-bg);border-color:var(--rev-line);color:var(--rev)}
.chip-crit{background:var(--crit-bg);border-color:var(--crit-line);color:var(--crit)}

/* section titles */
.sec-title{display:flex;align-items:center;gap:8px;font-family:var(--font-head);font-weight:700;color:var(--ink);font-size:1.02rem;margin:6px 0 10px}
.sec-count{font-family:var(--font-body);font-weight:700;font-size:.74rem;color:var(--muted);background:var(--surface);border:1px solid var(--line);padding:1px 8px;border-radius:999px}

/* finding cards (Streamlit keyed containers) */
div[class*="st-key-card-"]{background:var(--surface);border:1px solid var(--line)!important;border-radius:var(--radius);padding:14px 16px 6px 18px!important;box-shadow:var(--shadow);position:relative;transition:box-shadow .2s ease,border-color .2s ease}
div[class*="st-key-card-"]:hover{box-shadow:0 4px 14px rgba(15,23,42,.08);border-color:var(--line-2)!important}
div[class*="st-key-card-"]::before{content:"";position:absolute;left:0;top:10px;bottom:10px;width:4px;border-radius:0 4px 4px 0;background:var(--line-2)}
div[class*="st-key-card-critical"]::before{background:var(--crit)}
div[class*="st-key-card-review"]::before{background:var(--rev)}
div[class*="st-key-card-info"]::before{background:var(--inf)}
.f-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.f-type{display:inline-flex;align-items:center;gap:5px;font-size:.78rem;color:var(--muted)}
.f-risk{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:.78rem;color:var(--muted)}
.f-title{font-family:var(--font-head);font-weight:650;font-size:1.04rem;color:var(--ink);margin:8px 0 6px}
.f-why{margin:0 0 6px 0;padding-left:18px;color:var(--ink-2);font-size:.9rem;line-height:1.5}
.f-why li{margin:2px 0}
.verify{display:inline-flex;gap:6px;align-items:center;font-size:.8rem;color:var(--rev);background:var(--rev-bg);border:1px solid var(--rev-line);border-radius:6px;padding:3px 8px;margin:2px 0 8px}
.decision{display:flex;align-items:center;gap:8px;font-size:.84rem;padding:6px 10px;border-radius:8px;margin:4px 0 8px}
.decision.d-confirm{background:var(--ok-bg);color:var(--ok);border:1px solid var(--ok-line)}
.decision.d-override{background:var(--inf-bg);color:var(--inf);border:1px solid var(--inf-line)}
.decision small{color:var(--muted)}

/* meters */
.meter{display:inline-flex;align-items:center;gap:8px}
.meter-track{width:72px;height:6px;background:var(--line);border-radius:999px;overflow:hidden;display:inline-block}
.meter-fill{display:block;height:100%;border-radius:999px}
.m-hi{background:var(--ok)}.m-mid{background:#D97706}.m-lo{background:var(--crit)}
.meter-val{font-variant-numeric:tabular-nums;font-size:.78rem;color:var(--ink-2);min-width:30px}

/* medication table */
.medtab{width:100%;border-collapse:separate;border-spacing:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;font-size:.86rem;box-shadow:var(--shadow)}
.medtab th{text-align:left;font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);background:var(--surface-2);padding:9px 12px;border-bottom:1px solid var(--line);font-weight:700}
.medtab td{padding:9px 12px;border-bottom:1px solid var(--line);color:var(--ink-2);vertical-align:top}
.medtab tr:last-child td{border-bottom:0}
.medtab tr:hover td{background:#FAFCFE}
.med-name{font-weight:700;color:var(--ink)}
.med-raw{font-size:.76rem;color:var(--muted)}
.med-dose{font-size:.8rem;color:var(--ink-2);margin-top:3px}
.med-lists{margin-top:6px}.med-lists .chip{margin-bottom:0}
.st-active{color:var(--ok)}.st-held,.st-stopped{color:var(--rev)}.st-mentioned{color:var(--muted)}
.tabwrap{overflow-x:auto}

/* note viewer */
.note{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px 18px;font-family:var(--font-mono);font-size:.8rem;line-height:1.65;color:var(--ink-2);white-space:pre-wrap;max-height:640px;overflow:auto;box-shadow:var(--shadow)}
mark.d{background:var(--accent-soft);color:var(--accent);border-bottom:2px solid var(--accent);padding:0 1px;border-radius:2px;font-weight:700}
mark.d.hot{background:var(--crit-bg);color:var(--crit);border-bottom-color:var(--crit)}
.note mark.a{background:transparent;color:inherit;border-bottom:1px dashed var(--line-2)}
.legend{display:flex;gap:14px;font-size:.78rem;color:var(--muted);margin:0 0 8px}

/* empty / misc */
.empty{background:var(--surface);border:1px dashed var(--line-2);border-radius:var(--radius);padding:28px;text-align:center;color:var(--muted)}
.callout{display:flex;gap:10px;align-items:flex-start;background:var(--accent-soft);border:1px solid var(--accent-line);color:#155E75;border-radius:10px;padding:10px 12px;font-size:.86rem;margin-bottom:12px}
.callout.warn{background:var(--rev-bg);border-color:var(--rev-line);color:#92400E}
.stButton>button{border-radius:8px;font-weight:700;min-height:38px;transition:all .15s ease}
.stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent)}
.stButton>button[kind="primary"]:hover{background:#155E75;border-color:#155E75}
.stButton>button:focus-visible{outline:3px solid var(--accent-line);outline-offset:2px}
div[data-testid="stExpander"] details{border-radius:10px;border-color:var(--line)}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-family:var(--font-head);font-weight:600}
</style>
"""
