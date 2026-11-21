"""Prepare training and validation data for QLoRA fine-tuning in MLX chat format.

Output format per line (JSONL):
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "<clinical chunk text>"},
  {"role": "assistant", "content": "{\"meds\": [[\"drug\", \"strength\", ...]]}"}
]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import load_config
from athena.data import n2c2
from athena.extraction.chunking import chunk_text
from athena.extraction.corrections_fewshot import build_train_corrections_bank
from athena.extraction.llm_extractor import _prompt

COLS = ["Strength", "Dosage", "Form", "Route", "Frequency", "Duration"]


def format_note_chunks(notes, cfg: dict, system_prompt: str) -> list[dict]:
    items = []
    chunk_chars = cfg.get("chunk_chars", 4000)
    overlap = cfg.get("chunk_overlap_chars", 300)
    for note in notes:
        chunks = chunk_text(note.text, chunk_chars, overlap)
        for c in chunks:
            rows = []
            for drug in note.drugs():
                if c.start <= drug.start and drug.end <= c.end:
                    attrs = {
                        a.type: a.text
                        for a in note.attributes_of(drug.id)
                        if c.start <= a.start and a.end <= c.end
                    }
                    rows.append([drug.text] + [attrs.get(col, "") for col in COLS] + [""])
            if rows:
                items.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": c.text},
                        {"role": "assistant", "content": json.dumps({"meds": rows})},
                    ]
                })
    return items


def main():
    cfg = load_config()["extraction"]
    system = _prompt()
    out_dir = Path("data/processed/lora")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building training dataset from n2c2 train gold + train corrections...")
    train_notes = n2c2.load_split("train")
    train_items = format_note_chunks(train_notes, cfg, system)

    # Also incorporate snippet bank corrections
    corr_bank = build_train_corrections_bank(split="train")
    for snippet, rows in corr_bank:
        if rows:
            train_items.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": snippet},
                    {"role": "assistant", "content": json.dumps({"meds": rows})},
                ]
            })

    print(f"Total train examples: {len(train_items)}")
    train_path = out_dir / "train.jsonl"
    with train_path.open("w") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")
    print(f"Saved {train_path}")

    print("Building validation dataset from n2c2 val gold...")
    val_notes = n2c2.load_split("val")
    val_items = format_note_chunks(val_notes, cfg, system)
    print(f"Total val examples: {len(val_items)}")
    val_path = out_dir / "valid.jsonl"
    with val_path.open("w") as f:
        for it in val_items:
            f.write(json.dumps(it) + "\n")
    print(f"Saved {val_path}")


if __name__ == "__main__":
    main()
