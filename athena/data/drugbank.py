"""Loaders for the two DrugBank-derived DDI files.

- Kaggle extract: drug *names* + human-readable description (86 templates).
- jcsun-00 benchmark: DrugBank *IDs* + type label 0..85 + SMILES + standard splits.

Both come from the same DeepDDI DrugBank benchmark (README §3). A name<->ID
mapping is not included in either file; it will come from the DrugBank
open vocabulary.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from athena.config import data_path


def description_to_template(desc: str, drug1: str, drug2: str) -> str:
    """Replace the two drug names in a description with placeholders.

    Longer name first so that e.g. 'Iron' does not clobber 'Iron sucrose'.
    """
    out = desc
    for name, tag in sorted(((drug1, "{A}"), (drug2, "{B}")), key=lambda x: -len(x[0])):
        out = out.replace(name, tag)
    return out


@lru_cache(maxsize=None)
def load_kaggle_ddi() -> pd.DataFrame:
    """Columns: drug_a, drug_b, description, template."""
    df = pd.read_csv(data_path("drugbank_kaggle"))
    df.columns = ["drug_a", "drug_b", "description"]
    df["template"] = [
        description_to_template(d, a, b)
        for a, b, d in zip(df["drug_a"], df["drug_b"], df["description"])
    ]
    return df


@lru_cache(maxsize=None)
def load_benchmark_ddi() -> pd.DataFrame:
    """Columns: id_a, id_b, type (0..85). Negative-sample column dropped."""
    df = pd.read_csv(data_path("drugbank_benchmark") / "ddis.csv", usecols=["d1", "d2", "type"])
    return df.rename(columns={"d1": "id_a", "d2": "id_b"})


@lru_cache(maxsize=None)
def load_smiles() -> pd.DataFrame:
    return pd.read_csv(data_path("drugbank_benchmark") / "drug_smiles.csv")


def load_benchmark_split(setting: str, fold: str, part: str) -> pd.DataFrame:
    """e.g. load_benchmark_split('warm_start', 'fold0', 'train')."""
    return pd.read_csv(data_path("drugbank_benchmark") / setting / fold / f"{part}.csv")
