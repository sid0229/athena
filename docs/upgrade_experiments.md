# Upgrade experiments (external review suggestions)

A reviewer proposed a set of upgrades. Each was implemented behind a switch (or as a separate
experiment), measured against the existing system on the same data, and **kept only if it helped**.
Scripts and raw numbers are in `scripts/exp_*.py` and `docs/results/`.

## Summary

| Suggestion | Verdict | Evidence |
|---|---|---|
| Chemical-structure features (RDKit) in the XGBoost predictor | **Adopted** | Warm-start ROC-AUC 0.937 → 0.978, F1 0.859 → 0.925 |
| Realistically imbalanced evaluation | **Adopted** (reporting) | PR-AUC at 1:7 0.778 → 0.899, at 1:50 0.527 → 0.696 (graph → graph + chemistry) |
| Cold-start (unseen drug) evaluation and model | **Adopted** | One drug unseen: ROC-AUC 0.389 (graph) → 0.803 (cold-aware chemical model); both unseen 0.500 → 0.681 |
| Route awareness for topical / ophthalmic / otic / nasal drugs | **Adopted** | 35 irrelevant interaction findings removed on 38 notes; route-specific risks kept |
| Dynamic router (rules for lists, LLM for narrative) | Rejected | Saves ≤ 5% of LLM work before F1 drops; skipping the LLM entirely loses 0.017 F1 |
| Dynamic few-shot prompting (retrieved training examples) | Evaluated, not adopted (opt-in only) | Test: hybrid strict F1 0.794 → 0.804, but Drug F1 0.866 → 0.858 and LLM-only 0.659 → 0.632 |
| Semantic / vector search for drug-name normalisation | Rejected | Fixes typos but proposes dangerous wrong drugs at the same similarity (e.g. Sarna → tranylcypromine) |
| Dose-dependent downgrade rules | Not implemented | Downgrades are the unsafe direction; no labelled data to validate thresholds |
| ONC high-priority list as the only Critical anchor | Not implemented | See below |
| Graph database (KùzuDB) for multi-hop CYP reasoning | Not implemented | See below |
| SQLite WAL mode | Not needed | The interaction KB is opened read-only; WAL only helps concurrent writers |
| pyvis mechanism graph in the dashboard | Deferred | No enzyme/target data locally; UI-only, no measurable effect |
| Swap datasets (DrugCentral, MIMIC-IV-Note, TDC) | Partly applicable | See below |

## 1. XGBoost interaction predictor (`scripts/exp_ddi_upgrades.py`, `scripts/train_ddi_xgboost.py`)

Chemical features per pair: Morgan fingerprints (radius 2, 256 bits) as bit sum and bit product,
Tanimoto similarity, the highest similarity of each drug to the other drug's known partners, and
interactions recorded between the 5 chemically nearest neighbours of each drug. All computed from
training interactions only; the target edge is excluded (tested).

| DrugBank benchmark, fold 0 | Graph only | Chemistry only | Graph + chemistry |
|---|---|---|---|
| Warm start 1:1, ROC-AUC / F1 | 0.937 / 0.856 | 0.951 / 0.891 | **0.978 / 0.923** |
| Warm start 1:7 (benchmark density), PR-AUC | 0.775 | 0.781 | **0.895** |
| Warm start 1:50 (stress), PR-AUC | 0.525 | 0.460 | **0.697** |

The reviewer was right that graph features fail on unseen drugs, and in fact the graph model is
worse than chance there (ROC-AUC 0.39): it learns that drugs with many recorded interactions
interact, and a new drug has none. Simply adding chemistry did not fix this (0.53). What worked
was training a chemistry-only model **with one or both drugs' own interactions hidden**, the way
they are at test time:

| Cold start, 1:1 | Graph only | Graph + chemistry | Cold-aware chemistry |
|---|---|---|---|
| S2: one drug unseen, ROC-AUC | 0.389 | 0.532 | **0.802** |
| S1: both unseen, ROC-AUC | 0.500 | 0.466 | **0.678** |

On "1:1 inflates precision": true, and now reported. Note, however, that the benchmark itself is
13% dense (1 recorded interaction per 7 pairs), not 0.1%, because DrugBank records many low-grade
interactions. Precision falling with prevalence is arithmetic, not a model defect; it is one reason
the predictor's output is never shown as a verified flag.

Effect on the paper's specificity figure (same 200 held-out pairs): Athena hybrid specificity
0.95 → 0.97, sensitivity 0.79 → 0.89, balanced accuracy 0.87 → 0.93.

## 2. Route awareness (`scripts/exp_route_aware.py`)

Instead of an arbitrary "topical discount", Athena uses DDInter's own route-specific concepts
(210 of them, e.g. `timolol (ophthalmic)`, `lidocaine (topical)`). If the extracted Route/Form or
the drug text says topical, ophthalmic, otic or nasal, and such a concept exists, it is checked
instead of the systemic drug. Transdermal patches stay systemic (fentanyl, clonidine,
nitroglycerin) except local analgesics. Switch: `fusion.route_aware` in `configs/default.yaml`.

On 35 n2c2 test + 3 demo notes: 7 medications detected as local (miconazole powder, mupirocin,
clobetasol, hydrocortisone cream, camphor-menthol lotion, lidocaine patch), 35 findings removed
(1 Review: atorvastatin + topical miconazole; 34 Info), none added, Critical unchanged. Clinically
relevant local-route interactions are still flagged because DDInter records them on the route
concept: topical miconazole + warfarin (Major), ophthalmic timolol + salmeterol (Major).

## 3. Dynamic router (`scripts/exp_router.py`)

Each 4,000-character LLM chunk was scored by the share of list-like lines; chunks above a
threshold skipped the LLM (threshold tuned on validation, applied once to test).

| Test, 35 notes | Strict F1 | Chunks sent to the LLM |
|---|---|---|
| Current hybrid (no router) | 0.794 | 136 / 136 |
| Router, threshold 0.5 (chosen on validation) | 0.792 | 131 / 136 |
| Rules only (LLM never called) | 0.777 | 0 / 136 |

Discharge-summary chunks mix lists and narrative, so almost none are "pure list": the router saves
under 5% of LLM work, and any threshold that saves more loses F1 steadily. Not adopted.

## 4. Few-shot prompting with retrieved examples (`scripts/exp_fewshot.py`)

For each chunk, the 3 most similar 8-line snippets from the n2c2 *training* split (TF-IDF over
word 1–2-grams, offline) are inserted as worked examples with their gold rows. K = 3 was fixed in
advance (no tuning), so the validation result is not tuned on.

| Validation, 38 notes | Baseline prompt | Few-shot prompt |
|---|---|---|
| Hybrid, strict / lenient micro F1 | 0.792 / 0.870 | **0.801 / 0.884** |
| LLM only, strict / lenient micro F1 | 0.622 / 0.693 | **0.679 / 0.742** |
| LLM only, Dosage / Form F1 | 0.211 / 0.552 | **0.531 / 0.744** |
| Hybrid, Duration F1 | **0.547** | 0.454 |
| LLM time per note | ~1 min | ~1 min (2,270 s for 38 notes, under memory pressure) |

On validation the LLM itself improves a lot (+0.057 strict F1), mostly on Dosage and Form; in the
hybrid the gain is smaller (+0.009) because rules already cover those slots.

Confirmed once on the 35-note test subset (`docs/results/fewshot_test.json`):

| Test, 35 notes | Baseline prompt | Few-shot prompt |
|---|---|---|
| Hybrid, strict / lenient micro F1 | 0.794 / 0.878 | **0.804 / 0.880** |
| Hybrid, Drug F1 | **0.866** | 0.858 |
| Hybrid, Frequency F1 | 0.666 | **0.736** |
| Hybrid, Duration F1 | **0.485** | 0.421 |
| LLM only, strict F1 / Drug F1 | **0.659 / 0.768** | 0.632 / 0.691 |

The hybrid gain is consistent (+0.010) but comes almost entirely from Frequency, while Drug
detection — the input to every interaction check — gets slightly worse, and the LLM-only gain seen
on validation reverses on test. Not adopted as the default; it remains available as
`MedicationExtractor(..., fewshot=True)` (implemented in `athena/extraction/fewshot.py`).

## 5. Vector search for normalisation

Tested a character n-gram vector index over the 20,318-name lexicon on the 252 unmapped drug
mentions of the validation split. It fixes genuine typos (wafarin → warfarin, motelukast →
montelukast, verpamil → verapamil) but at the same similarity scores proposes wrong and dangerous
drugs: Sarna (anti-itch lotion) → tranylcypromine (an MAOI), Asacol → triamcinolone, Betadine →
bethanidine, glucose → cholecalciferol. No threshold separates the two groups, and a wrong mapping
creates false interactions. Most remaining unmapped mentions are drugs absent from both sources
(senna, guaifenesin) or class words (ACE-I, benzodiazepine), which no matcher can map. SapBERT
embeddings would need PyTorch on an 8 GB machine and carry the same failure mode; a safer use is
a "did you mean …?" suggestion shown to the pharmacist, never an automatic mapping.

## 6. Suggestions not implemented, and why

- **Dose-dependent downgrades** (e.g. Major → Minor below a dose). Downgrading is the unsafe
  direction, the thresholds need clinical sources per pair, and there is no labelled data to
  check them. Dose checks that *raise* attention could be added with the Batch 7 rule layer.
- **ONC high-priority list as the only Critical anchor.** The list (Phansalkar et al., 2012) is a
  small set of contraindicated pairs. Demoting everything else would push e.g. warfarin + aspirin
  out of Critical, and the list does not contain the combinations Athena currently under-rates
  (sertraline + tramadol, opioid + benzodiazepine). It fits as one cited input to Batch 7
  rules, not as a filter.
- **Graph database / multi-hop CYP inference.** Pair lookups already take 7 ms per patient. There
  is no enzyme or transporter data locally (the DrugBank academic XML is not used), and inferring
  undocumented interactions would *add* unverified alerts, against the alert-fatigue goal.
  Class-level rules (NSAID + ACE inhibitor) are planned in Batch 7 and work in SQLite.
- **Native RxNorm SCD mapping.** Its main benefit, route, is now handled with the extracted Route
  and DDInter's route concepts.
- **Datasets.** *DrugBank licence*: correct, academic use only; DrugCentral (open licence) is
  the right replacement if Athena is ever deployed, and is noted for future work. *MIMIC-IV-Note*:
  needs PhysioNet credentialing; recommended as external validation of extraction (the "rules fit
  n2c2's templates" concern is fair and can only be tested on another corpus). *TDC DDI*: its
  DrugBank task is the same DeepDDI data, and the cold-start splits used above already test the
  unseen-drug case. *Temporal drift*: checked 24 post-2010 drugs and brands; 19 map and are covered
  (e.g. empagliflozin, semaglutide, apixaban, Ozempic, Eliquis); missing are 2021+ drugs
  (tirzepatide, nirmatrelvir/Paxlovid, finerenone) and evolocumab.
- **Side finding:** combination brands such as Entresto (sacubitril/valsartan) normalise as
  "ambiguous" and are therefore not checked; worth fixing in the normaliser.
