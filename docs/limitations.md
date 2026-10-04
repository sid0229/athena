# Known limitations and data issues

Recorded as found. Each entry: what, where it matters, how Athena handles it.

## Data

### n2c2 2018 Track 2
- **Unofficial copy.** Obtained from `varunchaudharycs/biomedical_ner`, not the DBMI portal. Note counts match the official release (303 train / 202 test); the test split has 10,575 Drug entities. Official access still to be obtained before publication.
- **Validation split is not official.** `training_data_3` (38 notes) was held out by the repo author. Train (265) + val (38) = official 303 training notes.
- **Discontiguous entities.** 6,273 entity annotations have multiple spans. For 859 of them the annotated text includes the gap (e.g. `hydromorphone (Dilaudid` for spans `hydromorphone` + `Dilaudid`). *Handling:* character offsets are treated as authoritative; all contiguous entities match the note text exactly (tested).
- **Mislabelled relations.** 18 relations in train have a label that disagrees with the attribute entity type (14 `ADE-Drug` → Reason entity, 3 `Reason-Drug` → Drug entity, 1 `Strength-Drug` → Dosage entity). *Handling:* scoring uses entity types, not relation labels.

### MIMIC-III Clinical Database Demo
- **No clinical notes.** `NOTEEVENTS` is empty in the demo. Real note text comes only from n2c2 (itself drawn from MIMIC-III).
- **Drug names don't match DrugBank directly.** Only 183 / 592 distinct prescription names (35% of rows) match DrugBank names exactly (salt forms, abbreviations like `ns`/`d5w`, formulation suffixes). *Handling:* drug normaliser (Batch 2).

### DrugBank (Kaggle extract + jcsun-00 benchmark)
- **Subset, ~2017 snapshot.** 1,701–1,706 approved small-molecule drugs, ~191k pairs; current DrugBank is far larger.
- **No biologics.** e.g. heparin, insulin absent. *Handling:* DDInter added; KB reports `NOT_COVERED` rather than "no interaction".
- **No severity.** *Handling:* severity from DDInter; otherwise from a documented template→severity map, marked `derived`.
- **No name↔ID mapping between the two files.** Row orders differ, so they cannot be aligned. *Handling:* needs the DrugBank open vocabulary (CC0), which must be downloaded while logged in to DrugBank.

### DDInter
- **21% of pairs graded `Unknown`.** *Handling:* `Unknown` is never treated as safe; it has its own weight in fusion.
- Per-ATC files overlap (222,383 rows → 160,235 unique pairs). Severity levels are consistent across files (checked).
- **Not in the synopsis.** Declared in README §8.

### FAERS
- One quarter only (2026 Q2). Spontaneous reports: co-reporting is not causation. Used only as an optional, labelled signal.
