# Code review (November 2026): findings and fixes

A full review of the repository (extraction, normalisation and knowledge base, fusion and app server,
ML and evaluation, web UI) found the issues below. Each fix has a regression test where one is
possible (`pytest`: 133 tests). Numbers are on the same data as before unless stated.

## 1. Safety: missed or under-rated interactions

| Problem | Fix | Effect |
|---|---|---|
| The public DDInter download has only 8 of 14 ATC files, but a drug's mere presence in DDInter counted as "DDInter checked this pair", so pairs of drugs from missing classes (C, G, J, M, N, S) got a false "no known interaction" | Each concept stores its DDInter class file(s) (`ddinter_home`); DDInter covers a pair only if one drug's class file was downloaded (`checker._lookup_pair`) | lithium + hydrochlorothiazide / lisinopril now found |
| One drug split into two concepts across sources (lithium cation / lithium carbonate, potassium, magnesium, icosapent, loteprednol, prasterone, insulin aspart) | Reviewed merges in `kb_build.SALT_MERGES`, clinical name kept for display | lithium's 518 DrugBank records reachable again |
| Combination brands (Entresto, Percocet, Kaletra, Augmentin, … 563 RxNorm entries) were "ambiguous" and never checked | RxNorm multi-ingredient entries are mapped as combinations; the 57 genuinely ambiguous Wikidata aliases stay unchecked and are shown | specific-drug coverage on n2c2 test 87.3% → 88.0% |
| Template-derived severity is nearly always Moderate (held-out Major recall 8%); thin-evidence templates became "Unknown" (ranked below Minor) | `athena/verification/high_alert.py`: 11 reviewed, cited rules raise classic combinations to Major (MAOI + serotonergic, nitrate + PDE5i, opioid + benzodiazepine, lithium + diuretic/ACEi/ARB/NSAID, K-raising + ACEi/ARB, methotrexate + trimethoprim, strong CYP3A4 inhibitor + simvastatin/lovastatin/colchicine, tramadol + SSRI/SNRI, QT + QT, clopidogrel + omeprazole, allopurinol + thiopurine); thin templates → Moderate, or Major if ≥ 75% of ≥ 3 | fluoxetine + phenelzine, sildenafil + nitroglycerin, oxycodone + lorazepam, sertraline + tramadol now Major with the rule and source shown |
| Any "Word:" line ended a medication list; header variants were missed; a list with no following header ran to the end of the note | `athena/fusion/sections.py` rewritten: known headers only (singular, parentheses, all-caps), inline colon lines never end a list, prose paragraph ends it | +448 gold drugs inside discharge lists across 505 n2c2 notes; no list shortened wrongly |
| Without a discharge list every mention was "active" ("allergic to aspirin" → aspirin interaction) | Allergy sections and negated / historical mentions are not active; "held aspirin" attaches to aspirin; "not held" ignored | regression tests |
| Route-specific concepts are DDInter-only, so route awareness could hide interactions; route match was case-sensitive | If DDInter cannot speak for a route-specific pair, the systemic drugs are checked; routes lower-cased | ophthalmic timolol + verapamil found again |
| Normaliser: abbreviations applied after the index ("Percocet 5/325" ambiguous), frequency words not stripped, "levo" ambiguous, KCl in saline dropped | Abbreviations first, sig words stripped, "levo" removed, IV vehicle stripped first | |

## 2. Security and data integrity
- **Path traversal** in the server's web route (any project file readable, verified) → resolved path must stay inside `web/dist`; `TrustedHostMiddleware` blocks DNS rebinding.
- **Decisions recorded against a recomputed report** → the client sends the `report_hash` it viewed; the server keeps reports by hash and returns 409 instead of recomputing.
- Rules-fallback results (LLM offline) were cached as "hybrid" → not cached under the requested mode.
- **Audit log**: deleting the newest rows was undetectable → a sidecar anchor (row count + head hash) is checked by `verify()`, timestamps must not go backwards, appending to a broken chain is refused, connections are closed.
- CSV formula injection in exports (server and Streamlit) → cells starting with `= + - @` are escaped.

## 3. Results claims
- The DDI specificity figure called the XGBoost research model "Athena hybrid" → relabelled "XGBoost predictor (research)"; the footnote says live flags come from the database lookup.
- The cautious-prompt LLM result was not reproducible → both prompts are scripted in `scripts/eval_specificity.py`, the 200 pairs are frozen in `docs/results/specificity_pairs.json`, benchmark negatives that are interactions in Athena's own KB are flagged, Wilson 95% intervals added.
- XGBoost: early stopping on validation data (cold: the benchmark's `val.csv`), threshold chosen on validation, 3 folds with mean ± SD, bootstrap CIs, a drug-popularity baseline, real test prevalence reported (not "1:1").
- Extraction: the evaluation script can no longer silently fall back to rules; bootstrap CIs over notes and a paired hybrid-vs-rules test are reported.

| DDI predictor, after the review | Result |
|---|---|
| Warm start, 3 folds, ROC-AUC | **0.981 ± 0.000** (fold 0: 95% CI 0.980–0.982; PR-AUC 0.985) |
| Graph-only / logistic regression / drug-popularity baseline | 0.932 / 0.911 / 0.80 |
| 1 : 7 and 1 : 50 imbalance, PR-AUC | 0.914 / 0.734 |
| Unseen drugs, 3 folds (S2 / S1) | 0.81 ± 0.01 / 0.71 ± 0.02 |
| LLM vs predictor on the frozen 200 pairs (balanced accuracy) | plain prompt 0.50, cautious prompt 0.56, XGBoost **0.94** (both LLM results reproduced exactly by the script) |

Early stopping still runs to the 2,000-tree cap on the warm model (validation loss keeps falling slowly);
the gain from 600 to 2,000 trees is +0.003 ROC-AUC.

## 4. Extraction
- Form regex: "Tablet, Delayed Release (E.C.)" was cut to "Tablet" (alternation order) → long forms first.
- Frequency spans could not cross a line break inside brackets ("BID (2 times a⏎day)").
- Added "p.o.", "oral", "q.d.", "q. day", "at night".
- LLM cache: refuses to reuse outputs made with a different prompt file or options; no pointless retry at temperature 0; a failed chunk falls back to rules for that chunk only; cache warm-up survives a timeout.

| n2c2 test, 35 notes, strict micro F1 (95% CI) | Before | After |
|---|---|---|
| Athena hybrid | 0.794 | **0.814** (0.791–0.834) |
| Rules only | 0.777 | 0.799 (0.783–0.816) |
| LLM only | 0.659 | 0.665 (0.635–0.695) |
| Hybrid − rules (paired bootstrap) | | +0.015 (−0.005 to +0.034): not significant on 35 notes |

**Full test set (202 notes, run by Rupesh):** before the fixes hybrid 0.796 / rules 0.782; **with the fixes
hybrid 0.817 (95% CI 0.806–0.827) / rules 0.806**; hybrid − rules +0.011 strict (paired 95% CI +0.003 to
+0.018) — significant.

Tried and **not adopted** (no measurable gain on validation): keeping LLM numeric dosages (+0.0002),
detecting therapeutic oxygen (+0.0002), restricting typo matching to curated names (lost Keflex,
Versed, Thorazine).

## 5. New: alert-level evaluation
`scripts/eval_alerts.py` runs the checker on n2c2 *gold* drugs (discharge list) and compares with the
full pipeline's alerts on the 35 test notes: pair precision **0.999**, pair recall **0.938**, all **27/27**
gold Major pairs found and placed in an actionable tier, all 139 actionable alerts correct. (Same KB on
both sides, so this measures extraction, sections and status — not the KB.)

## 6. ML signal for uncovered pairs
For pairs no database covers, the Triage page now shows the XGBoost probability as a clearly labelled,
unverified research signal (warm model for known drugs, cold-start model otherwise). It is not part of
the report, the risk tiers or the audit log.

## 7. Web UI
Med Diff no longer shows "Unchanged" when a dose is missing, or "Omission" for rows with no finding;
omission colours match Triage; decision failures are shown; late responses cannot overwrite a newer
analysis; J/K move a visible focus; light-mode green/orange pass AA; the source note is a native modal
dialog; labelled controls; Cmd/Ctrl+B ignored while typing; the high-alert rule is shown as evidence.

## 8. Not done
- **86-class interaction-type prediction** (for comparison with DeepDDI-style papers): not needed by
  Athena — flags show the DrugBank sentence / DDInter grade, never an ML-predicted type. A first
  LightGBM attempt diverged on rare classes; left as future work.

## 9. Removed
- `uncertainty_penalty` risk formula: clinically inverted (typical-confidence Minor interactions became
  Critical) and only compared by tier counts; `scripts/compare_risk_formulas.py` deleted (its saved
  figure and JSON are kept as a record).
- Dead imports and an unused alias.
