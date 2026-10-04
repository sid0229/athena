"""Check a medication list from the command line.

    python scripts/check_meds.py "coumadin 5 mg" "ASA 81" "Lasix 40" "Percocet" "tylenol"
    python scripts/check_meds.py --all "..."     # also list NO_KNOWN / NOT_COVERED pairs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.verification.checker import InteractionChecker, Medication, PairStatus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meds", nargs="+", help='medication strings; suffix ":held" or ":stopped" to mark status')
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    meds = []
    for m in args.meds:
        name, _, status = m.partition(":")
        meds.append(Medication(name, status or "active"))
    r = InteractionChecker().check(meds)

    print("Medications")
    for med, n in r.medications:
        print(f"  {med.name:30s} [{med.status}] -> {', '.join(n.drugs) or '-'}  ({n.status}, {n.stage}, {n.score})")
    if r.duplicates:
        print("\nDuplicate therapy")
        for d in r.duplicates:
            print(f"  {d.concept.name}: " + " + ".join(m.name for m in d.medications))
    print(f"\nInteractions ({len(r.interactions)})")
    for p in r.interactions:
        print(f"  [{p.severity:8s}] {p.a.name} + {p.b.name}   basis={p.severity_basis} sources={','.join(p.sources)}")
        for m in p.mechanisms:
            print(f"       - {m.text}  (template: {m.derived_severity}, {m.pct_major}% Major of {m.n_overlap_graded})")
    if args.all:
        for st in (PairStatus.NO_KNOWN_INTERACTION, PairStatus.NOT_COVERED):
            print(f"\n{st.value}")
            for p in r.pairs:
                if p.status is st:
                    print(f"  {p.a.name} + {p.b.name}  checked={p.checked_sources}")
    if r.unchecked:
        print("\nNot checked (shown to pharmacist)")
        for med, n in r.unchecked:
            print(f"  {med.name} ({n.status})")
    print(f"\nPairs: {len(r.pairs)} | interaction {r.count(PairStatus.INTERACTION)} | "
          f"no known {r.count(PairStatus.NO_KNOWN_INTERACTION)} | not covered {r.count(PairStatus.NOT_COVERED)}")


if __name__ == "__main__":
    main()
