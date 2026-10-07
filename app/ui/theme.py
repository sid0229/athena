"""Athena visual system: design tokens, local fonts, icons and small HTML helpers.

Direction: a calm, hospital-grade clinical workstation. Navy + white + teal + neutral
grey; semantic colours only where clinically meaningful and always paired with an
icon and a text label. Figtree for headings, Atkinson Hyperlegible for body text,
both served from app/static (no network requests). All colours, type sizes,
spacing, radii and shadows are defined once as CSS custom properties in `css()`.
"""

from __future__ import annotations

import base64
import html
import re

# Lucide icons (ISC licence), drawn with a CSS mask so they take the current text colour
# (Streamlit's HTML sanitiser strips inline <svg>).
_ICON_PATHS = {
    "octagon-alert": '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "edit-note": '<path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c0-1.1.9-2 2-2h2"/><path d="M4 8c0-1.1.9-2 2-2h2"/>',
    "repeat": '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "arrow-right-left": '<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>',
    "circle-help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "user": '<circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 0 0-16 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
}

TIERS = {
    "critical": {"label": "Critical", "icon": "octagon-alert"},
    "review": {"label": "Review", "icon": "triangle-alert"},
    "info": {"label": "Info", "icon": "info"},
}
TYPE_LABELS = {
    "INTERACTION": ("Drug interaction", "arrow-right-left"),
    "DUPLICATE_THERAPY": ("Duplicate therapy", "copy"),
    "OMISSION": ("Omission", "circle-help"),
    "NEW_MEDICATION": ("New medication", "plus"),
    "DOSE_CHANGE": ("Dose change", "repeat"),
    "NOT_CHECKED": ("Not checked", "circle-help"),
    "NOT_COVERED": ("Not covered", "circle-help"),
}
_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "dr.", "mr.", "mrs.", "ms.", "prof."}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _svg_uri(name: str, stroke: float) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def icon(name: str, size: int = 16, stroke: float = 2, cls: str = "") -> str:
    uri = _svg_uri(name, stroke)
    return (f'<span class="ic {cls}" aria-hidden="true" style="width:{size}px;height:{size}px;'
            f'-webkit-mask:url({uri}) center/contain no-repeat;mask:url({uri}) center/contain no-repeat"></span>')


def tier_badge(tier: str) -> str:
    t = TIERS[tier]
    return f'<span class="badge tier-{tier}">{icon(t["icon"], 13, 2.4)}<span>{t["label"]}</span></span>'


def chip(text: str, kind: str = "neutral", ic: str | None = None) -> str:
    return f'<span class="chip chip-{kind}">{icon(ic, 12) if ic else ""}{esc(text)}</span>'


def meter(value: float, label: str | None = None) -> str:
    pct = max(0, min(100, round(value * 100)))
    lvl = "hi" if value >= 0.85 else "mid" if value >= 0.6 else "lo"
    return (f'<span class="meter" role="img" aria-label="{esc(label or "confidence")} {pct}%">'
            f'<span class="meter-track"><span class="meter-fill m-{lvl}" style="width:{pct}%"></span></span>'
            f'<span class="meter-val">{value:.2f}</span></span>')


def initials(name: str) -> str:
    """'Dr. A. Sharma' -> 'AS'; 'Mr. Bhatt' -> 'MB'; 'Priya' -> 'PR'."""
    words = [w for w in re.split(r"[\s,]+", name.strip()) if w]
    if not words:
        return "?"
    title = words[0] if words[0].lower() in _TITLES else None
    rest = [w for w in words if w.lower() not in _TITLES and not re.fullmatch(r"[A-Za-z]{2,}D", w)] or words
    if len(rest) >= 2:
        return (rest[0][0] + rest[-1][0]).upper()
    if title:
        return (title[0] + rest[0][0]).upper()
    return rest[0][:2].upper()


def avatar(name: str, size: str = "md") -> str:
    return f'<span class="avatar avatar-{size}" aria-hidden="true">{esc(initials(name))}</span>'


def section_title(text: str, count: int | None = None, sub: str | None = None) -> str:
    c = f'<span class="sec-count">{count}</span>' if count is not None else ""
    s = f'<span class="sec-sub">{esc(sub)}</span>' if sub else ""
    return f'<div class="sec-title"><span>{esc(text)}</span>{c}{s}</div>'


def label(text: str) -> str:
    return f'<div class="field-label">{esc(text)}</div>'


def css() -> str:
    return """
<style>
@font-face{font-family:'Figtree';src:url('app/static/fonts/Figtree-var.woff2') format('woff2');font-weight:300 900;font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('app/static/fonts/AtkinsonHyperlegible-400.woff2') format('woff2');font-weight:400;font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('app/static/fonts/AtkinsonHyperlegible-700.woff2') format('woff2');font-weight:700;font-display:swap}

/* ===================================================================== tokens */
:root{
  /* surfaces */
  --bg:#F5F8FA; --surface:#FFFFFF; --surface-soft:#F8FAFB;
  --navy-950:#081525; --navy-900:#0D1B2E; --navy-800:#142640; --navy-700:#1B3554;
  /* brand */
  --teal-700:#086F82; --teal-600:#0B8296; --teal-500:#1599AE; --teal-100:#E6F4F6; --teal-50:#F1FAFB;
  /* text — --text-muted is decorative only (4.0:1); metadata uses --text-secondary (6.2:1) */
  --text-primary:#152238; --text-secondary:#50627A; --text-muted:#718198;
  --border:#D7E1E8; --border-soft:#E8EEF2;
  /* status */
  --success:#267A5C; --success-soft:#EAF6F0;
  --warning:#B87916; --warning-soft:#FFF5DF;
  --danger:#B94A48;  --danger-soft:#FCEDEC;
  /* clinical tiers (icon + label + colour, never colour alone) */
  --crit:#9E3D3B; --crit-bg:#FCEDEC; --crit-line:#E9B9B7;
  --rev:#9B650F;  --rev-bg:#FFF5DF;  --rev-line:#EBCF98;
  --inf:#246A80;  --inf-bg:#EAF4F7;  --inf-line:#BFDCE5;
  /* type */
  --font-head:'Figtree',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --font-body:'Atkinson Hyperlegible',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --font-mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  --fs-title:32px; --fs-section:22px; --fs-card:18px; --fs-body:16px; --fs-small:14px; --fs-meta:13px; --fs-eyebrow:12px;
  --lh:1.55;
  /* spacing scale */
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:20px; --s6:24px; --s8:32px; --s10:40px; --s12:48px;
  /* shape */
  --r-card:10px; --r-input:8px; --r-btn:8px; --r-badge:6px;
  --shadow:0 1px 3px rgba(13,27,46,.06);
}

/* ===================================================================== base */
.stApp{background:var(--bg);color:var(--text-primary)}
.stApp,.stApp p,.stApp label,.stApp input,.stApp textarea,.stApp li,.stMarkdown,[data-testid="stMarkdownContainer"]{
  font-family:var(--font-body);font-size:var(--fs-body);line-height:var(--lh)}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded'!important}
h1,h2,h3,h4{font-family:var(--font-head)!important;color:var(--text-primary)}
.block-container{padding-top:var(--s8);padding-bottom:var(--s12);max-width:1440px}
header[data-testid="stHeader"]{background:transparent;height:0}
.ic{display:inline-block;vertical-align:-3px;flex:none;background:currentColor}
:focus-visible{outline:3px solid var(--teal-500)!important;outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
.stHtml,[data-testid="stHtml"]{width:100%}

/* ===================================================================== sidebar */
section[data-testid="stSidebar"]{background:var(--navy-950);border-right:1px solid var(--navy-800);width:300px!important}
section[data-testid="stSidebar"] *{color:#D5DEE8}
[data-testid="stSidebarHeader"]{padding-bottom:var(--s2)}
[data-testid="stLogo"]{height:38px!important;max-width:100%}
[data-testid="stSidebarNav"]{border-bottom:1px solid var(--navy-800);padding-bottom:var(--s3);margin-bottom:var(--s2)}
[data-testid="stSidebarNav"] a{border-radius:var(--r-input);font-size:15px}
[data-testid="stSidebarNav"] a span{font-size:15px}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:var(--navy-800)}
[data-testid="stSidebarNav"] a:hover{background:var(--navy-900)}
section[data-testid="stSidebar"] [data-baseweb="select"]>div{background:var(--navy-900)!important;border-color:var(--navy-700)!important;border-radius:var(--r-input)}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#EEF3F8!important}
.side-label{font-family:var(--font-body);font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;
  color:#8FA2B8!important;font-weight:700;margin:var(--s4) 0 var(--s2)}
.side-div{height:1px;background:var(--navy-800);margin:var(--s4) 0 0}
.id-card{display:flex;gap:var(--s3);align-items:center;background:var(--navy-900);border:1px solid var(--navy-700);
  border-radius:var(--r-card);padding:var(--s3)}
.id-name{font-family:var(--font-head);font-weight:700;font-size:15px;color:#F4F7FA!important;line-height:1.25}
.id-role{font-size:var(--fs-meta);color:#A9B8C9!important;line-height:1.3}
.id-state{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#9FD8C2!important;margin-top:2px}
.id-empty .id-name{color:#F4D7A8!important}
.avatar{display:inline-grid;place-items:center;border-radius:50%;font-family:var(--font-head);font-weight:700;flex:none;
  background:var(--teal-700);color:#FFFFFF!important;letter-spacing:.02em}
.avatar-md{width:40px;height:40px;font-size:15px}
.avatar-sm{width:30px;height:30px;font-size:12px}
.avatar-empty{background:var(--navy-700)}
.status{display:flex;gap:var(--s3);align-items:flex-start;padding:6px 0}
.status .dot{margin-top:7px}
.status-title{font-size:14px;font-weight:700;color:#EEF3F8!important;line-height:1.35}
.status-sub{font-size:var(--fs-meta);color:#A9B8C9!important;line-height:1.35}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.dot-ok{background:#3FB58A}.dot-off{background:#E0A43A}
.side-foot{font-size:var(--fs-meta);color:#93A5B9!important;line-height:1.5;margin-top:var(--s5);
  border-top:1px solid var(--navy-800);padding-top:var(--s3)}
section[data-testid="stSidebar"] [data-testid="stPopoverButton"],
section[data-testid="stSidebar"] .stPopover button{background:transparent!important;border:1px solid var(--navy-700)!important;
  color:#D5DEE8!important;min-height:34px;font-size:14px}
section[data-testid="stSidebar"] .stPopover button:hover{border-color:var(--teal-500)!important}

/* ===================================================================== page header */
.eyebrow{font-size:var(--fs-eyebrow);letter-spacing:.1em;text-transform:uppercase;color:var(--teal-700);font-weight:700}
.page-title{font-family:var(--font-head);font-size:var(--fs-title);font-weight:720;color:var(--text-primary);
  margin:var(--s1) 0 var(--s2);letter-spacing:-.02em;line-height:1.2}
.page-sub{color:var(--text-secondary);font-size:var(--fs-body);max-width:760px;line-height:1.5}
.meta-line{display:flex;flex-wrap:wrap;gap:var(--s2) var(--s4);color:var(--text-secondary);font-size:var(--fs-small);margin-top:var(--s1)}
.meta-line .ic{color:var(--text-muted)}
.who{display:flex;justify-content:flex-end;align-items:center;gap:var(--s3)}
.who-label{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--text-secondary);font-weight:700;text-align:right}
.who-name{font-family:var(--font-head);font-weight:700;font-size:15px;color:var(--text-primary);line-height:1.25;text-align:right}
.who-role{font-size:var(--fs-meta);color:var(--text-secondary);text-align:right}
.who-none{font-size:var(--fs-small);color:var(--rev);text-align:right}
.hash{font-family:var(--font-mono);font-size:var(--fs-meta);color:var(--text-secondary)}
.rule{height:1px;background:var(--border);margin:var(--s5) 0}

/* ===================================================================== tabs */
.stTabs [data-baseweb="tab-list"]{gap:var(--s6);border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{font-family:var(--font-head);font-weight:600;font-size:15px;color:var(--text-secondary);
  padding:var(--s3) 2px;min-height:44px;background:transparent}
.stTabs [data-baseweb="tab"] p{font-size:15px;font-family:var(--font-head);font-weight:600}
.stTabs [aria-selected="true"],.stTabs [aria-selected="true"] p{color:var(--teal-700)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--teal-700)!important;height:2px}
.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-panel"]{padding-top:var(--s5)}

/* ===================================================================== inputs & buttons */
.field-label{font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;color:var(--text-secondary);
  font-weight:700;margin:0 0 var(--s2)}
.stTextArea textarea,.stTextInput input{border-radius:var(--r-input)!important;font-size:15px!important}
.stTextArea textarea{font-family:var(--font-mono)!important;font-size:14px!important;line-height:1.6!important}
[data-baseweb="select"]>div{border-radius:var(--r-input)!important}
.stButton>button,.stDownloadButton>button,.stPopover button{border-radius:var(--r-btn);font-weight:650;font-size:15px;
  min-height:42px;transition:background .15s ease,border-color .15s ease}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--teal-700)!important;
  border-color:var(--teal-700)!important;color:#fff!important}
.stButton>button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{background:#075E6E!important;border-color:#075E6E!important}
.stButton>button[kind="primary"]:disabled,[data-testid="stBaseButton-primary"]:disabled{background:#B9CCD3!important;border-color:#B9CCD3!important}
.stButton>button[kind="secondary"],[data-testid="stBaseButton-secondary"]{background:var(--surface);border-color:var(--border);color:var(--text-primary)}
.stButton>button[kind="secondary"]:hover{border-color:var(--teal-600);color:var(--teal-700)}

/* ===================================================================== panels / context */
div[class*="st-key-panel"]{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);
  padding:var(--s5) var(--s5) var(--s4)!important;box-shadow:var(--shadow)}
.ctx-name{font-family:var(--font-head);font-weight:700;font-size:20px;color:var(--text-primary);line-height:1.25}
.ctx-sub{font-size:var(--fs-small);color:var(--text-secondary)}
.doc{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-input);padding:var(--s4) var(--s5);
  font-family:var(--font-mono);font-size:14px;line-height:1.7;color:#24324A;white-space:pre-wrap;overflow:auto}
.doc-preview{max-height:300px}
.doc-full{max-height:620px}
.doc-head{font-family:var(--font-body);font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;
  color:var(--text-secondary);font-weight:700;margin-bottom:var(--s2)}
.action-meta{display:flex;flex-wrap:wrap;gap:var(--s2) var(--s4);align-items:center;color:var(--text-secondary);font-size:var(--fs-small);
  min-height:42px}
.notice{display:flex;gap:var(--s3);align-items:flex-start;background:var(--teal-100);border:1px solid #CBE6EA;color:#0A5462;
  border-radius:var(--r-input);padding:var(--s3) var(--s4);font-size:var(--fs-small);line-height:1.5}
.notice b{display:block;font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;margin-bottom:2px}
.notice .ic{margin-top:2px}
.notice.warn{background:var(--warning-soft);border-color:#F0D9A6;color:#6E4A0B}
.notice.ok{background:var(--success-soft);border-color:#BFE3D2;color:#1D5E46}
.notice.bad{background:var(--danger-soft);border-color:var(--crit-line);color:#7C302F}

/* ===================================================================== KPI strip */
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-card);box-shadow:var(--shadow);overflow:hidden}
.kpi{padding:var(--s4) var(--s5);border-right:1px solid var(--border-soft)}
.kpi:last-child{border-right:0}
.kpi-label{display:flex;align-items:center;gap:6px;font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;
  font-weight:700;color:var(--text-secondary)}
.kpi-value{font-family:var(--font-head);font-size:26px;font-weight:700;color:var(--text-primary);line-height:1.15;
  margin-top:var(--s1);font-variant-numeric:tabular-nums}
.kpi-foot{font-size:var(--fs-meta);color:var(--text-secondary)}
.k-critical .kpi-label{color:var(--crit)}.k-review .kpi-label{color:var(--rev)}.k-info .kpi-label{color:var(--inf)}
.k-zero .kpi-value{color:var(--text-muted)}
@media (max-width:1100px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpi:nth-child(2){border-right:0}}

/* ===================================================================== section titles */
.sec-title{display:flex;align-items:center;gap:var(--s3);font-family:var(--font-head);font-weight:700;color:var(--text-primary);
  font-size:var(--fs-section);margin:var(--s2) 0 var(--s4);letter-spacing:-.01em}
.sec-count{font-family:var(--font-body);font-weight:700;font-size:var(--fs-meta);color:var(--text-secondary);background:var(--surface);
  border:1px solid var(--border);padding:1px var(--s2);border-radius:999px}
.sec-sub{font-family:var(--font-body);font-weight:400;font-size:var(--fs-small);color:var(--text-secondary)}

/* ===================================================================== badges & chips */
.badge{display:inline-flex;align-items:center;gap:6px;font-weight:700;font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  padding:3px 8px;border-radius:var(--r-badge);border:1px solid}
.tier-critical{color:var(--crit);background:var(--crit-bg);border-color:var(--crit-line)}
.tier-review{color:var(--rev);background:var(--rev-bg);border-color:var(--rev-line)}
.tier-info{color:var(--inf);background:var(--inf-bg);border-color:var(--inf-line)}
.badge-neutral{color:var(--text-secondary);background:var(--surface-soft);border-color:var(--border)}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:var(--fs-meta);font-weight:700;padding:2px 8px;border-radius:var(--r-badge);
  border:1px solid var(--border);background:var(--surface-soft);color:var(--text-secondary);margin:0 6px 6px 0;white-space:nowrap}
.chip-accent{background:var(--teal-50);border-color:#C7E5EA;color:var(--teal-700)}
.chip-warn{background:var(--rev-bg);border-color:var(--rev-line);color:var(--rev)}

/* ===================================================================== finding cards */
div[class*="st-key-card-"]{background:var(--surface);border:1px solid var(--border)!important;border-radius:var(--r-card);
  padding:var(--s4) var(--s5) var(--s3) var(--s6)!important;box-shadow:var(--shadow);position:relative;gap:var(--s2)!important}
div[class*="st-key-card-"]::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:var(--r-card) 0 0 var(--r-card);background:var(--border)}
div[class*="st-key-card-critical"]::before{background:var(--crit)}
div[class*="st-key-card-review"]::before{background:var(--rev)}
div[class*="st-key-card-info"]::before{background:var(--inf)}
.f-head{display:flex;align-items:center;gap:var(--s3);flex-wrap:wrap}
.f-type{display:inline-flex;align-items:center;gap:6px;font-size:var(--fs-small);color:var(--text-secondary)}
.f-sev{margin-left:auto;font-family:var(--font-head);font-weight:700;font-size:15px}
.f-sev.sev-major{color:var(--crit)}.f-sev.sev-moderate{color:var(--rev)}.f-sev.sev-minor,.f-sev.sev-unknown{color:var(--inf)}
.f-sev small{font-family:var(--font-body);font-weight:400;font-size:var(--fs-meta);color:var(--text-secondary)}
.f-title{font-family:var(--font-head);font-weight:700;font-size:var(--fs-card);color:var(--text-primary);margin:var(--s3) 0 var(--s1);line-height:1.3}
.f-summary{font-size:15px;color:#2E3D55;margin:0 0 var(--s3)}
.f-label{font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;color:var(--text-secondary);font-weight:700;margin:var(--s2) 0 6px}
.f-why{margin:0;padding-left:20px;color:#2E3D55;font-size:15px;line-height:1.55}
.f-why li{margin:2px 0}
.f-foot{display:flex;flex-wrap:wrap;gap:var(--s2) var(--s5);align-items:center;font-size:var(--fs-meta);color:var(--text-secondary);
  border-top:1px solid var(--border-soft);padding-top:var(--s3);margin-top:var(--s3)}
.f-foot b{color:var(--text-primary);font-variant-numeric:tabular-nums}
.verify{display:inline-flex;gap:6px;align-items:center;font-size:var(--fs-small);color:var(--rev);background:var(--rev-bg);
  border:1px solid var(--rev-line);border-radius:var(--r-badge);padding:3px 8px;margin:var(--s1) 0 var(--s2)}
.decision{display:flex;align-items:center;gap:var(--s2);font-size:var(--fs-small);padding:var(--s2) var(--s3);border-radius:var(--r-input);margin:var(--s1) 0}
.decision.d-confirm{background:var(--success-soft);color:var(--success);border:1px solid #BFE3D2}
.decision.d-override{background:var(--surface-soft);color:var(--text-primary);border:1px solid var(--border)}
.decision small{color:var(--text-secondary);margin-left:auto}
.override-hint{font-size:var(--fs-meta);color:var(--text-secondary)}

/* finding card v2: collapsed by default */
div[class*="st-key-card-"]{padding:var(--s5) var(--s5) var(--s2) var(--s6)!important}
.f-title{font-size:19px!important;margin:var(--s3) 0 6px!important}
.f-summary{font-size:16px!important;color:#2E3D55;margin:0 0 var(--s2)!important;line-height:1.5}
.f-sev{font-family:var(--font-body)!important;font-size:12px!important;letter-spacing:.08em;text-transform:uppercase;
  display:inline-flex;align-items:baseline;gap:6px}
.f-sev small{text-transform:none;letter-spacing:0}
.f-stats{display:flex;align-items:center;gap:var(--s2);font-size:var(--fs-small);color:var(--text-secondary);margin:2px 0 var(--s2)}
.f-stats b{color:var(--text-primary);font-variant-numeric:tabular-nums}
.f-stats .sep{color:var(--text-muted)}
.f-history{display:flex;align-items:center;gap:6px;font-size:var(--fs-meta);color:var(--text-secondary);
  background:var(--surface-soft);border:1px solid var(--border-soft);border-radius:var(--r-badge);padding:4px 10px;margin:2px 0 var(--s2)}
.f-history-ts{margin-left:auto;color:var(--text-secondary);font-size:12px}
div[class*="st-key-card-"] [data-testid="stExpander"] details{border:0!important;border-top:1px solid var(--border-soft)!important;
  border-radius:0!important;background:transparent!important;margin-top:var(--s1)}
div[class*="st-key-card-"] [data-testid="stExpander"] summary{padding:var(--s2) 0!important}
div[class*="st-key-card-"] [data-testid="stExpander"] summary p{font-family:var(--font-body)!important;font-size:14px!important;
  font-weight:700!important;color:var(--teal-700)!important}
div[class*="st-key-card-"] [data-testid="stExpanderDetails"]{padding:0 0 var(--s2)!important}
.ev-grid{display:grid;grid-template-columns:1fr;gap:var(--s3)}
.ev-block{background:var(--surface-soft);border:1px solid var(--border-soft);border-radius:var(--r-input);padding:var(--s3) var(--s4)}
.ev-block .f-label{margin-top:0}
.ev-list{margin:var(--s1) 0 0;padding-left:18px;font-size:15px;color:#2E3D55;line-height:1.5}
.ev-list li{margin:3px 0}
.ev-sub{color:var(--text-secondary);font-size:var(--fs-meta)}
.ev-text{font-size:15px;color:#2E3D55}
.ev-table{width:100%;border-collapse:collapse;font-size:var(--fs-small)}
.ev-table td{padding:4px 8px 4px 0;color:#2E3D55;vertical-align:top}
.ev-table td:last-child{color:var(--text-secondary)}

/* workflow progress */
.progress-wf{display:flex;align-items:center;gap:var(--s5);flex-wrap:wrap}
.pw-text{font-size:var(--fs-small);color:var(--text-secondary);display:flex;align-items:baseline;gap:6px;flex-wrap:wrap}
.pw-num{font-family:var(--font-head);font-size:20px;font-weight:700;color:var(--text-primary);font-variant-numeric:tabular-nums}
.pw-help{color:var(--text-secondary);font-size:var(--fs-meta);margin-left:var(--s2)}
.pw-track{flex:1;min-width:180px;max-width:420px;height:8px;background:var(--border-soft);border-radius:999px;overflow:hidden}
.pw-fill{height:100%;background:var(--teal-600);border-radius:999px}

/* reconciliation v2 */
.rc-head{display:flex;justify-content:space-between;padding:var(--s3) var(--s4);background:var(--surface-soft);border-bottom:1px solid var(--border);
  font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:var(--text-secondary)}
.rc-row{display:flex;justify-content:space-between;align-items:flex-start;gap:var(--s3);padding:var(--s3) var(--s4);border-bottom:1px solid var(--border-soft)}
.rc-row:last-of-type{border-bottom:0}
.rc-main{min-width:0}
.rc-row .rc-name{font-family:var(--font-head);font-size:16px;font-weight:700;color:var(--text-primary);line-height:1.3}
.rc-row .rc-dose{font-size:var(--fs-small);color:var(--text-secondary);margin-top:2px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.rc-row .rc-dose .ic{color:var(--text-muted)}
.rc-gone{color:var(--crit);font-weight:700}
.rc-row .rc-badge{flex:none;margin-top:2px}
.rc-same{font-size:var(--fs-meta);color:var(--text-secondary)}

/* ===================================================================== final polish */
/* Streamlit artifacts */
[data-testid="stDecoration"],[data-testid="stMainMenu"],#MainMenu,footer{display:none!important}
[data-testid="stToolbar"]{right:var(--s4)}
.gap-4{height:var(--s4)}.gap-6{height:var(--s6)}
/* account chip */
.account{display:inline-flex;align-items:center;gap:var(--s3);background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-card);padding:var(--s2) var(--s4) var(--s2) var(--s2);box-shadow:var(--shadow)}
.account .who-label,.account .who-name,.account .who-role,.account .who-none{text-align:left}
.account-empty{padding:var(--s3) var(--s4);color:var(--rev)}
.account-empty .who-none{color:var(--rev);font-size:var(--fs-small);font-weight:700}
/* medication names */
.rc-row .rc-name,.med-name{font-size:18px!important}
.rc-same{display:inline-flex;align-items:center;gap:5px}
/* cards: hover + calm transitions */
div[class*="st-key-card-"]{transition:border-color .15s ease,box-shadow .15s ease}
div[class*="st-key-card-"]:hover{border-color:#C5D3DC!important;box-shadow:0 2px 6px rgba(13,27,46,.07)}
/* form controls */
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--teal-600)!important;box-shadow:0 0 0 3px var(--teal-100)!important}
[data-baseweb="select"]>div:focus-within{border-color:var(--teal-600)!important;box-shadow:0 0 0 3px var(--teal-100)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--surface-soft);border:1.5px dashed var(--border)!important;border-radius:var(--r-card);
  padding:var(--s6)!important}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--teal-600)!important;background:var(--teal-50)}
[data-testid="stFileUploaderDropzoneInstructions"] span{font-size:15px;color:var(--text-primary)}
[data-testid="stPopoverBody"]{border-radius:var(--r-card)!important;border:1px solid var(--border)!important;
  box-shadow:0 8px 24px rgba(13,27,46,.12)!important}
div[data-testid="stExpander"] summary:hover p{color:var(--teal-700)}
.stTabs [data-baseweb="tab"]:hover p{color:var(--text-primary)}
/* consistent icon/baseline alignment in meta rows */
.meta-line span,.action-meta span,.f-type,.f-stats{display:inline-flex;align-items:center;gap:6px}
@media (min-width:1700px){.block-container{max-width:1480px}}

/* inline override panel (inside the finding card) */
div[class*="st-key-ovpanel-"]{background:#F7FAFB;border:1px solid var(--border);border-radius:var(--r-input);
  padding:var(--s4) var(--s5)!important;margin:var(--s3) 0 var(--s2);gap:var(--s3)!important}
div[class*="st-key-ovpanel-"] label p{font-size:15px!important;font-weight:700;color:var(--text-primary)}
div[class*="st-key-ovpanel-"] [data-baseweb="select"]>div{background:var(--surface);min-height:44px}
div[class*="st-key-ovpanel-"] [data-baseweb="select"] div{font-size:15px;white-space:normal;overflow:visible;text-overflow:clip}
div[class*="st-key-ovpanel-"] textarea{font-family:var(--font-body)!important;font-size:15px!important;background:var(--surface)!important}
.ov-title{display:flex;align-items:center;gap:var(--s2);font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;
  font-weight:700;color:var(--text-primary)}
.ov-sub{font-size:var(--fs-small);color:var(--text-secondary);margin-top:2px}
.ov-sub b{color:var(--text-primary)}
.ov-help{font-size:var(--fs-meta);color:var(--text-secondary)}
@media (max-width:900px){div[class*="st-key-ovpanel-"] [data-testid="stHorizontalBlock"]{flex-direction:column;align-items:stretch}}

/* reconciliation empty state (discharge list only) */
div[class*="st-key-panel-recon"]{gap:var(--s3)!important}
.rs-count{font-family:var(--font-head);font-size:18px;font-weight:700;color:var(--text-primary);display:flex;align-items:baseline;gap:var(--s2)}
.rs-num{font-size:26px;font-variant-numeric:tabular-nums}
.rs-ok{display:flex;align-items:center;gap:6px;font-size:var(--fs-small);color:var(--success);margin-top:2px}
.rs-div{height:1px;background:var(--border-soft);margin:var(--s4) 0 var(--s3)}
.rs-label{display:flex;align-items:center;gap:6px;color:var(--teal-700)}
.rs-text{font-size:15px;color:#2E3D55;line-height:1.55}
.rs-muted{font-size:var(--fs-meta);color:var(--text-secondary)}
.rs-card{display:flex;gap:var(--s3);align-items:flex-start;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-card);padding:var(--s5);box-shadow:var(--shadow);color:var(--teal-700)}
.rs-title{font-family:var(--font-head);font-weight:700;font-size:16px;color:var(--text-primary);margin-bottom:2px}
div[class*="st-key-panel-recon"] textarea{font-family:var(--font-body)!important;font-size:15px!important}

/* ===================================================================== meters */
.meter{display:inline-flex;align-items:center;gap:var(--s2)}
.meter-track{width:56px;height:5px;background:var(--border-soft);border-radius:999px;overflow:hidden;display:inline-block}
.meter-fill{display:block;height:100%;border-radius:999px}
.m-hi{background:#6FA898}.m-mid{background:#D2A24C}.m-lo{background:#C77B79}
.meter-val{font-variant-numeric:tabular-nums;font-size:var(--fs-meta);color:var(--text-secondary);min-width:30px}

/* ===================================================================== reconciliation */
.recon{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);box-shadow:var(--shadow);overflow:hidden}
.recon-head,.recon-row{display:grid;grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr);gap:var(--s2);align-items:center;
  padding:var(--s3) var(--s4)}
.recon-head{background:var(--surface-soft);border-bottom:1px solid var(--border);font-size:var(--fs-eyebrow);letter-spacing:.09em;
  text-transform:uppercase;font-weight:700;color:var(--text-secondary)}
.recon-row{border-bottom:1px solid var(--border-soft)}
.recon-row:last-child{border-bottom:0}
.rc-name{font-weight:700;font-size:15px;color:var(--text-primary);line-height:1.3}
.rc-dose{font-size:var(--fs-meta);color:var(--text-secondary)}
.rc-none{font-size:var(--fs-small);color:var(--text-secondary);font-style:italic}
.rc-arrow{color:var(--text-muted);text-align:center}
.rc-arrow.gone{color:var(--crit)}
.rc-badge{margin-top:6px}
.recon-foot{padding:var(--s2) var(--s4);font-size:var(--fs-meta);color:var(--text-secondary);background:var(--surface-soft);border-top:1px solid var(--border)}

/* ===================================================================== medication table */
.medtab{width:100%;border-collapse:separate;border-spacing:0;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-card);overflow:hidden;box-shadow:var(--shadow)}
.medtab th{text-align:left;font-size:var(--fs-eyebrow);letter-spacing:.09em;text-transform:uppercase;color:var(--text-secondary);
  background:var(--surface-soft);padding:var(--s3) var(--s4);border-bottom:1px solid var(--border);font-weight:700}
.medtab td{padding:var(--s3) var(--s4);border-bottom:1px solid var(--border-soft);color:#2E3D55;vertical-align:top;font-size:15px}
.medtab tr:last-child td{border-bottom:0}
.medtab td.c{text-align:center}
.med-name{font-family:var(--font-head);font-weight:700;font-size:16px;color:var(--text-primary)}
.med-raw{font-size:var(--fs-meta);color:var(--text-secondary)}
.med-dose{font-size:var(--fs-small);color:#2E3D55;margin-top:2px}
.yes{color:var(--teal-700)}.no{color:var(--text-muted)}
.st-pill{display:inline-flex;align-items:center;gap:6px;font-size:var(--fs-small);font-weight:700}
.st-active{color:var(--success)}.st-held,.st-stopped{color:var(--rev)}.st-mentioned{color:var(--text-secondary)}
.kv{width:100%;border-collapse:collapse;font-size:var(--fs-small)}
.kv td{padding:var(--s2) 0;border-bottom:1px solid var(--border-soft);vertical-align:top;color:#2E3D55}
.kv td:first-child{color:var(--text-secondary);width:42%;padding-right:var(--s3)}
.kv tr:last-child td{border-bottom:0}

/* ===================================================================== source note */
mark.d{background:var(--teal-100);color:#07505E;border-bottom:2px solid var(--teal-600);padding:0 2px;border-radius:2px;font-weight:700}
mark.d.hot{background:var(--crit-bg);color:var(--crit);border-bottom-color:var(--crit)}
.legend{display:flex;gap:var(--s4);font-size:var(--fs-meta);color:var(--text-secondary);margin:0 0 var(--s2)}
.legend mark{padding:0 6px}

/* ===================================================================== audit timeline */
.timeline{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);box-shadow:var(--shadow);overflow:hidden}
.tl-day{background:var(--surface-soft);border-bottom:1px solid var(--border);padding:var(--s2) var(--s5);font-size:var(--fs-eyebrow);
  letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:var(--text-secondary)}
.tl-row{display:grid;grid-template-columns:72px 40px minmax(0,1fr);gap:var(--s3);padding:var(--s4) var(--s5);border-bottom:1px solid var(--border-soft)}
.tl-row:last-child{border-bottom:0}
.tl-time{font-variant-numeric:tabular-nums;font-weight:700;color:var(--text-primary);font-size:15px}
.tl-time small{display:block;font-weight:400;color:var(--text-secondary);font-size:12px}
.tl-who{font-weight:700;color:var(--text-primary);font-size:15px}
.tl-act{font-size:15px;color:#2E3D55;margin-top:2px}
.tl-act .verb-confirm{color:var(--success);font-weight:700}
.tl-act .verb-override{color:var(--rev);font-weight:700}
.tl-reason{font-size:var(--fs-small);color:var(--text-secondary);margin-top:2px}
.tl-meta{font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);margin-top:6px}

/* ===================================================================== misc */
.empty{background:var(--surface);border:1px dashed var(--border);border-radius:var(--r-card);padding:var(--s8);text-align:center;color:var(--text-secondary)}
div[data-testid="stExpander"] details{border-radius:var(--r-card);border-color:var(--border);background:var(--surface)}
div[data-testid="stExpander"] summary p{font-family:var(--font-head);font-weight:600;font-size:15px}
[data-testid="stProgress"]>div>div>div>div{background:var(--teal-600)}
.about-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--s4);margin:var(--s5) 0}
.about-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);padding:var(--s5);box-shadow:var(--shadow)}
.about-card h4{font-size:17px;margin:var(--s2) 0 var(--s2)}
.about-card p{font-size:15px;color:#2E3D55;margin:0}
.about-card .ic{color:var(--teal-700)}
@media (max-width:1100px){.about-grid{grid-template-columns:1fr}}
</style>
"""
