"""Loader for MIMIC-III Clinical Database Demo v1.4 tables used by Athena."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from athena.config import data_path

PRESCRIPTION_COLS = [
    "subject_id", "hadm_id", "icustay_id", "startdate", "enddate", "drug_type",
    "drug", "drug_name_generic", "ndc", "prod_strength",
    "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp", "route",
]


@lru_cache(maxsize=None)
def load_prescriptions() -> pd.DataFrame:
    df = pd.read_csv(
        data_path("mimic_demo") / "PRESCRIPTIONS.csv",
        usecols=PRESCRIPTION_COLS,
        dtype={"ndc": "string", "dose_val_rx": "string", "form_val_disp": "string"},
        parse_dates=["startdate", "enddate"],
    )
    # Generic name when present, otherwise the order name.
    df["drug_name"] = df["drug_name_generic"].fillna(df["drug"]).str.strip()
    return df


def admission_med_lists(min_drugs: int = 1) -> dict[int, list[str]]:
    """Distinct drug names per hospital admission (hadm_id -> sorted names)."""
    df = load_prescriptions()
    lists = (
        df.assign(name=df["drug_name"].str.lower())
        .groupby("hadm_id")["name"]
        .agg(lambda s: sorted(set(s.dropna())))
    )
    return {int(h): meds for h, meds in lists.items() if len(meds) >= min_drugs}


@lru_cache(maxsize=None)
def load_table(name: str) -> pd.DataFrame:
    """Any other demo table by name, e.g. load_table('ADMISSIONS')."""
    return pd.read_csv(data_path("mimic_demo") / f"{name.upper()}.csv")
