"""Build data/processed/drug_lexicon.json and print a MIMIC coverage report."""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.data import mimic  # noqa: E402
from athena.normalize.lexicon import build_lexicon, load_lexicon, save_lexicon  # noqa: E402
from athena.normalize.normalizer import DrugNormalizer  # noqa: E402


def main():
    lex = build_lexicon()
    save_lexicon(lex)
    load_lexicon.cache_clear()
    print(f"canonical drugs        : {len(lex['canonical'])}")
    print(f"with DrugBank ID       : {len(lex['drugbank_id'])}")
    print(f"Kaggle names -> ID     : {len(lex['kaggle_name_to_id'])}")
    print(f"surface strings        : {len(lex['index'])}  {dict(collections.Counter(lex['source'].values()))}")

    norm = DrugNormalizer()
    counts = collections.Counter(mimic.load_prescriptions()["drug_name"].str.lower())
    res = {name: norm.normalize(name) for name in counts}
    by_status = collections.Counter()
    for name, r in res.items():
        by_status[r.status] += counts[name]
    total = sum(counts.values())
    drug_rows = total - by_status["non_drug"]
    print(f"\nMIMIC-demo PRESCRIPTIONS ({total} rows, {len(counts)} names)")
    for k, v in by_status.most_common():
        print(f"  {k:10s} {v:6d} rows ({v / total:.1%})")
    print(f"  mapped among drug rows (excl. fluids/supplies): {by_status['mapped'] / drug_rows:.1%}")
    print("  stages:", dict(collections.Counter(r.stage for r in res.values())))
    top = sorted(((counts[n], n) for n, r in res.items() if r.status == "unmapped"), reverse=True)[:25]
    print("  top unmapped:", top)


if __name__ == "__main__":
    main()
