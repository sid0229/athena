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
- **Drug names don't match DrugBank directly.** Only 183 / 592 distinct prescription names (35% of rows) match DrugBank names exactly. *Resolved in Batch 2a:* normaliser maps 91% of drug rows (fluids/supplies excluded).

### DrugBank (Kaggle extract + jcsun-00 benchmark)
- **Subset, ~2017 snapshot.** 1,701–1,706 approved small-molecule drugs, ~191k pairs; current DrugBank is far larger.
- **No biologics.** e.g. heparin, insulin absent. *Handling:* DDInter added; KB reports `NOT_COVERED` rather than "no interaction".
- **No severity.** *Handling:* severity from DDInter; otherwise from a documented template→severity map, marked `derived`.
- **No name↔ID mapping between the two files** (row orders differ). *Resolved in Batch 2a:* Wikidata label/alias match + interaction-graph matching maps 1,678 / 1,701 names; mapped pairs agree with the benchmark >99.9%.

### DDInter
- **21% of pairs graded `Unknown`.** *Handling:* `Unknown` is never treated as safe; it has its own weight in fusion.
- Per-ATC files overlap (222,383 rows → 160,235 unique pairs). Severity levels are consistent across files (checked).
- **Not in the synopsis.** Declared in README §8.

### FAERS
- One quarter only (2026 Q2). Spontaneous reports: co-reporting is not causation. Used only as an optional, labelled signal.

## Drug-name normalisation (Batch 2a)
- **Coverage is not complete.** ~13% of specific-drug mentions in n2c2 remain unmapped (e.g. senna, guaifenesin, kayexalate, polyethylene glycol — mostly drugs absent from both interaction sources). Unmapped drugs are shown to the pharmacist as "not checked", never dropped.
- **Abbreviation list was tuned on n2c2 *train* only.** Test-split coverage (87.3%) is reported without having inspected test mentions.
- **Wikidata aliases can be wrong.** Audit found and fixed: "sodium polystyrene sulfonate" → tolevamer, "albumin human" → perflutren, "megestrol" → nomegestrol (fuzzy). Mitigations: ID-based alias linking, RxCUI consistency filter, explicit blocklist, strict fuzzy rule.
- **Some KB entries are duplicates of one drug** (e.g. `fluticasone` vs `fluticasone propionate`, insulin variants). To be merged by DrugBank ID when building the KB (Batch 2b).
- **26 DrugBank-extract drugs have no DrugBank ID** (not on Wikidata, not resolvable via the interaction graph). They work by name in the KB; they lack SMILES for the Phase 2 similarity model.

## Interaction KB (Batch 2b)
- **Route-specific drugs are not yet route-aware.** DDInter separates e.g. `timolol (ophthalmic)` from systemic `timolol`; the normaliser currently strips "ophthalmic"/"topical" and maps to the systemic concept. This **over-flags** (safe direction). Planned: use the extracted Route to choose the route-specific concept.
- **Derived severity is coarse.** A DrugBank template is only a weak predictor of severity (most templates are mixed; e.g. "risk or severity of adverse effects" is 20% Major). Derived grades are labelled `derived` and shown with their evidence; DDInter grades always take precedence.
- **24 templates have too little overlap to derive severity** (< 10 graded pairs) → `Unknown`, which is never treated as safe.
- **Very high raw hit rate.** ~52% of all drug pairs in a MIMIC-demo admission have some KB record (DrugBank is inclusive). Tiering, not the KB, must control alert volume.
- **Salt merges are a reviewed allow-list** (12 pairs). Four RxNorm salt→parent links were rejected as clinically different drugs (isosorbide mononitrate, choline salicylate, trastuzumab deruxtecan, radiolabelled dotatate).

## Checker (Batch 3)
- **Some clinically Major pairs are graded Moderate.** Example: oxycodone + lorazepam (opioid + benzodiazepine, FDA boxed warning) exists only in DrugBank under the generic "risk or severity of adverse effects" template (19.7% Major) → derived Moderate. Pinned by `test_known_limitation_opioid_benzo_not_major`. Possible fix: a small, cited rule set for boxed-warning classes (would be a declared addition).
- **DDInter `Unknown` dominates the volume** (~half of interactions per MIMIC patient). These are real records with no grade; fusion must place them below graded Major/Moderate without hiding them.
- **Medication lists in MIMIC are reconstructed** from PRESCRIPTIONS start/end dates (drugs active on the last day), not from a true discharge list.
