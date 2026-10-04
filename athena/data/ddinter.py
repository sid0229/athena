"""Loader for DDInter pairwise interaction files (one CSV per ATC code)."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from athena.config import data_path

SEVERITY_ORDER = ("Major", "Moderate", "Minor", "Unknown")


@lru_cache(maxsize=None)
def load_ddinter() -> pd.DataFrame:
    """Unique unordered pairs with severity.

    Columns: id_a, drug_a, id_b, drug_b, level, atc_codes.
    The per-ATC files overlap (a pair is listed under each drug's class);
    pairs are de-duplicated with id_a < id_b. Levels are consistent across
    files (checked in Batch 1), so no conflict resolution is needed.
    """
    frames = []
    for f in sorted(data_path("ddinter").glob("ddinter_downloads_code_*.csv")):
        df = pd.read_csv(f)
        df["atc"] = f.stem.rsplit("_", 1)[-1]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["id_a", "drug_a", "id_b", "drug_b", "level", "atc"]

    swap = df["id_a"] > df["id_b"]
    df.loc[swap, ["id_a", "drug_a", "id_b", "drug_b"]] = df.loc[
        swap, ["id_b", "drug_b", "id_a", "drug_a"]
    ].to_numpy()

    out = (
        df.groupby(["id_a", "drug_a", "id_b", "drug_b", "level"], as_index=False)["atc"]
        .agg(lambda s: "".join(sorted(set(s))))
        .rename(columns={"atc": "atc_codes"})
    )
    out["level"] = pd.Categorical(out["level"], categories=SEVERITY_ORDER, ordered=True)
    return out


def ddinter_drugs() -> pd.DataFrame:
    """One row per DDInter drug: ddinter_id, name."""
    df = load_ddinter()
    drugs = pd.concat([
        df[["id_a", "drug_a"]].set_axis(["ddinter_id", "name"], axis=1),
        df[["id_b", "drug_b"]].set_axis(["ddinter_id", "name"], axis=1),
    ])
    return drugs.drop_duplicates().sort_values("ddinter_id", ignore_index=True)
