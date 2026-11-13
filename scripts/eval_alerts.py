"""Alert-level evaluation: does the whole pipeline raise the interaction alerts it should?

    python scripts/eval_alerts.py            # 35 cached n2c2 test notes, hybrid (cache only)

Reference alerts come from the n2c2 *gold* drug annotations: gold drug mentions inside the
discharge medication list (all gold mentions if the note has none) are normalised and checked
with the same InteractionChecker. The system's alerts come from the full pipeline (extraction,
sections, status, normalisation, checker, fusion). Because both sides use the same checker and
knowledge base, the differences measure what extraction, section parsing and status inference
lose or add — not the knowledge base itself. No human labels are needed.

Reports pair-level precision / recall overall, recall of Major pairs, and the share of gold
Major pairs that reach an actionable tier (Critical or Review). Writes docs/results/alerts_eval.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze, local_route  # noqa: E402
from athena.fusion.sections import medication_sections  # noqa: E402
from athena.verification.checker import InteractionChecker, Medication  # noqa: E402


@dataclass
class _GoldMed:
    raw: str
    attributes: dict
    kb_names: list


def gold_pairs(note, checker):
    secs = medication_sections(note.text)
    dis = [s for s in secs if s.kind == "discharge"]
    drugs = [d for d in note.drugs() if not dis or any(s.start <= d.start < s.end for s in dis)]
    meds = []
    for d in drugs:
        routes = [note.entities[r.arg1].text for r in note.relations if r.arg2 == d.id and r.type == "Route-Drug"]
        forms = [note.entities[r.arg1].text for r in note.relations if r.arg2 == d.id and r.type == "Form-Drug"]
        norm = checker.normalizer.normalize(d.text)
        gm = _GoldMed(raw=d.text, attributes={"Route": " ".join(routes), "Form": " ".join(forms)}, kb_names=list(norm.drugs))
        meds.append(Medication(d.text, route=local_route(gm)))
    rep = checker.check(meds)
    return {frozenset((p.a.concept_id, p.b.concept_id)): p.severity for p in rep.interactions}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    cfg = load_config()["extraction"]
    llm = OllamaExtractor()
    llm.available = lambda: True  # cache only, never a silent rules fallback

    def _no_call(text):
        raise RuntimeError("uncached chunk")
    llm._call = _no_call
    notes = [n for n in n2c2.load_split("test")[: args.limit]
             if all(llm._cache_path(c.text).exists() for c in chunk_text(n.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]))]
    checker = InteractionChecker()
    ex = MedicationExtractor("hybrid", llm=llm, normalizer=checker.normalizer)
    tot = {"gold": 0, "sys": 0, "tp": 0, "gold_major": 0, "major_found": 0, "major_actionable": 0, "sys_actionable": 0,
           "actionable_correct": 0}
    misses, extras = [], []
    for n in notes:
        gold = gold_pairs(n, checker)
        r = analyze(n.text, ex, checker)
        sys_ = {frozenset(f.id[4:].split("|")): f for f in r.findings if f.type == "INTERACTION"}
        tp = set(gold) & set(sys_)
        tot["gold"] += len(gold); tot["sys"] += len(sys_); tot["tp"] += len(tp)
        for k, sev in gold.items():
            if sev == "Major":
                tot["gold_major"] += 1
                if k in sys_:
                    tot["major_found"] += 1
                    tot["major_actionable"] += sys_[k].tier in ("critical", "review")
                else:
                    misses.append((n.id, sorted(k)))
        for k, f in sys_.items():
            if f.tier in ("critical", "review"):
                tot["sys_actionable"] += 1
                tot["actionable_correct"] += k in gold
                if k not in gold:
                    extras.append((n.id, f.title, f.tier))
    pr = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    alerts_per_note = round(tot["sys_actionable"] / len(notes), 2) if notes else 0.0
    res = {
        "notes": len(notes), "counts": tot,
        "pair_precision": pr(tot["tp"], tot["sys"]), "pair_recall": pr(tot["tp"], tot["gold"]),
        "major_recall": pr(tot["major_found"], tot["gold_major"]),
        "major_reaching_actionable_tier": pr(tot["major_actionable"], tot["gold_major"]),
        "actionable_alert_precision": pr(tot["actionable_correct"], tot["sys_actionable"]),
        "alerts_per_note": alerts_per_note,
        "missed_major_examples": misses[:15], "extra_actionable_examples": extras[:15],
        "reference": "checker run on n2c2 gold drug mentions in the discharge list (same KB and rules)",
    }
    for k in ("pair_precision", "pair_recall", "major_recall", "major_reaching_actionable_tier", "actionable_alert_precision", "alerts_per_note"):
        print(f"{k:32s} {res[k]}")
    print(tot)
    if args.save:
        (ROOT / "docs/results/alerts_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
