"""Run the full Athena pipeline on a note and print the report.

    python scripts/run_pipeline.py --n2c2 val:130153
    python scripts/run_pipeline.py path/to/note.txt [--mode rules] [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.data import n2c2  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--n2c2", help="split:note_id")
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--json")
    ap.add_argument("--all", action="store_true", help="also print info-tier findings")
    args = ap.parse_args()

    if args.n2c2:
        split, nid = args.n2c2.split(":")
        text = next(n.text for n in n2c2.load_split(split) if n.id == nid)
    else:
        text = Path(args.path).read_text()

    r = analyze(text, MedicationExtractor(args.mode))
    c = r.meta["counts"]
    print(f"mode={r.meta['extraction_mode']}  {r.meta['extraction_seconds']}s  hash={r.report_hash[:12]}")
    for w in r.meta["warnings"]:
        print("WARNING:", w)
    print(f"medications {c['medications']} (active {c['active']}) | pairs {c['pairs_checked']} | "
          f"interactions {c['interactions']} | tiers {c['tiers']}")
    print("\nActive medications")
    for m in r.medications:
        if m.status == "active":
            attrs = " ".join(m.attributes.get(k, "") for k in ("Strength", "Route", "Frequency")).strip()
            print(f"  {m.name:32s} {attrs:35s} conf={m.confidence:.2f}  [{m.norm_status}]")
    for tier in ("critical", "review", "info"):
        fs = [f for f in r.findings if f.tier == tier]
        if not fs or (tier == "info" and not args.all):
            if fs:
                print(f"\n{tier.upper()}: {len(fs)} (use --all to show)")
            continue
        print(f"\n{tier.upper()} ({len(fs)})")
        for f in fs:
            print(f"  [{f.risk:.2f}] {f.title}" + ("  ⚠ verify" if f.needs_verification else ""))
            for line in f.explanation:
                print(f"         {line}")
    if args.json:
        Path(args.json).write_text(json.dumps(r.to_dict(), indent=1))


if __name__ == "__main__":
    main()
