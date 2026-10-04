"""Build data/processed/interactions.db and print a summary + MIMIC coverage report."""

import itertools
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from athena.config import data_path  # noqa: E402
from athena.data import mimic  # noqa: E402
from athena.normalize.normalizer import DrugNormalizer  # noqa: E402
from athena.verification.kb_build import build_kb  # noqa: E402


def main():
    stats = build_kb()
    for k, v in stats.items():
        print(f"{k:22s}: {v}")

    with sqlite3.connect(data_path("kb_db")) as con:
        tsev = pd.read_sql("SELECT * FROM template_severity", con)
        print("\nDerived template severity:", tsev["derived_severity"].value_counts().to_dict())
        print(tsev[tsev["derived_severity"] == "Major"][["template", "n_overlap_graded", "pct_major"]].to_string(index=False))

        # Coverage on MIMIC-demo admissions: concept-level pairs per admission.
        names = dict(con.execute("SELECT name, concept_id FROM names"))
        pairs = {(a, b) for a, b in con.execute("SELECT DISTINCT a, b FROM interactions")}
    norm = DrugNormalizer()
    adm_pairs = adm_hits = 0
    per_adm = []
    for hadm, meds in mimic.admission_med_lists().items():
        concepts = set()
        for m in meds:
            r = norm.normalize(m)
            if r.status == "mapped":
                concepts.update(names[d] for d in r.drugs)
        cp = [tuple(sorted(p)) for p in itertools.combinations(sorted(concepts), 2)]
        hits = sum(p in pairs for p in cp)
        adm_pairs += len(cp)
        adm_hits += hits
        per_adm.append(hits)
    s = pd.Series(per_adm)
    print(f"\nMIMIC-demo admissions: {len(s)}, drug pairs checked: {adm_pairs}, "
          f"pairs with a KB interaction: {adm_hits} ({adm_hits / adm_pairs:.1%})")
    print(f"interacting pairs per admission: median {s.median():.0f}, max {s.max()}")


if __name__ == "__main__":
    main()
