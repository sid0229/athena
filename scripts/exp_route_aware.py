"""Experiment: route-aware checking (topical/ophthalmic/otic/nasal -> DDInter route-specific concept).

    python scripts/exp_route_aware.py

Runs the full pipeline on the 35 cached n2c2 test notes and the 3 synthetic demo notes with
route awareness off and on, and lists every interaction finding that changes.
Writes docs/results/route_aware_experiment.json.
"""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import ROOT, load_config  # noqa: E402
from athena.fusion.report import analyze, local_route  # noqa: E402
from athena.verification.checker import InteractionChecker  # noqa: E402
from compare_risk_formulas import notes  # noqa: E402


def main():
    docs, ex = notes()
    checker = InteractionChecker()
    base = load_config()["fusion"]
    tiers = {k: Counter() for k in ("off", "on")}
    changes, routed = [], []
    for name, text in docs:
        res = {}
        for k, flag in (("off", False), ("on", True)):
            cfg = copy.deepcopy(base)
            cfg["route_aware"] = flag
            r = analyze(text, ex, checker, fusion_cfg=cfg)
            res[k] = r
            tiers[k].update(f.tier for f in r.findings if f.type == "INTERACTION")
        for e in res["on"].medications:
            if e.status == "active" and (rt := local_route(e)):
                routed.append({"note": name, "drug": e.raw, "route": rt, "attrs": e.attributes})
        off = {f.title: f for f in res["off"].findings if f.type == "INTERACTION"}
        on = {f.title: f for f in res["on"].findings if f.type == "INTERACTION"}
        for t in sorted(set(off) | set(on)):
            a, b = off.get(t), on.get(t)
            if not a or not b or a.tier != b.tier:
                changes.append({"note": name, "finding": t, "off": a.tier if a else None, "on": b.tier if b else None})
    summary = {
        "notes": len(docs),
        "tiers_off": dict(tiers["off"]), "tiers_on": dict(tiers["on"]),
        "local_route_medications": routed,
        "changed_findings": changes,
    }
    for k in ("off", "on"):
        print(k, dict(tiers[k]))
    print(f"{len(routed)} active meds detected with a local route:")
    for r in routed:
        print("  ", r["note"], "|", r["drug"], "->", r["route"], "|", r["attrs"])
    print(f"{len(changes)} findings changed:")
    for c in changes:
        print("  ", c)
    (ROOT / "docs/results/route_aware_experiment.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
