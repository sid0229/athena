# Athena — Team Roadmap

Internal working document: progress, decisions, open questions and future work.
The judge-facing overview is [`README.md`](README.md). Known limitations are in [`docs/limitations.md`](docs/limitations.md).

| | |
|---|---|
| Mid-term evaluation | **8 Oct 2026** (demo) |
| Final evaluation | **24 Nov 2026** (buffer for report / paper into early December) |
| Status | Phase 1 complete (Batches 0–6) + XGBoost interaction predictor (Batch 11 preview) · 114 tests passing |

---

## 1. Progress log — Phase 1 (4–7 Oct 2026)

| Batch | What was built | Key result |
|---|---|---|
| **0 — Setup** | Python 3.11 venv (uv), pinned requirements, Ollama + `llama3.2:3b`, config, smoke tests, git | Model + scispaCy run locally; ~3.8 GB peak RAM on M2 8 GB |
| **1 — Data** | Loaders for n2c2 2018 (from zips), MIMIC-III demo, DrugBank extract + benchmark, DDInter | Counts pinned in tests; data issues logged in `docs/limitations.md` |
| **2a — Normaliser** | Drug lexicon from DrugBank, DDInter, Wikidata (linked by DrugBank ID) and RxNorm; ordered deterministic matcher | 87% of specific n2c2 drug mentions, 91% of MIMIC drug rows mapped; 3 false merges found by audit and fixed |
| **2b — Interaction KB** | SQLite KB: 2,421 drug concepts, 311,187 pairs; severity for DrugBank-only pairs derived from DDInter overlap | 10 templates derived Major (e.g. QTc 93%, serotonergic 91%) |
| **3 — Checker** | Every active pair → interaction / no known / not covered; duplicates, unchecked, inactive | 7 ms per MIMIC patient; median 49 interacting pairs but ~1 Major per patient |
| **4 — Extraction** | Local LLM (compact JSON rows) + grounding guard + slot repair + rules ensemble; 4 modes | n2c2 held-out test (35 notes): micro F1 .794 strict / .878 lenient (val .792 — no overfitting) |
| **5 — Fusion** | Sections, medication list + status, admission-vs-discharge reconciliation, risk × tiers, explanations, report hash | Catches omissions, new drugs, dose changes; Major never below Review |
| **6 — Dashboard** | Streamlit review UI, confirm/override with reasons, hash-chained append-only audit log, 3 synthetic demo patients | Runs fully offline at `localhost:8501` |
| **UI rounds (7 Oct)** | Clinical workstation redesign; reviewer identity card; progressive-disclosure finding cards; inline override panel; reconciliation empty state with "Add prior medication list"; Athena logo | Frontend only — backend untouched; regression-tested headless |
| **11-preview — XGBoost DDI predictor (7 Oct)** | Shtar-style graph-similarity features + XGBoost, DrugBank benchmark warm-start split; logistic-regression baseline | Test F1 .859, ROC-AUC .937 (LR baseline F1 .843, ROC-AUC .911) |

### Honest findings worth reporting
- The **rules ensemble alone (.777)** nearly matches **hybrid (.794)** on n2c2's regular medication lists; the LLM adds most on drug names in narrative text and status words.
- **48%** of LLM attribute values are rejected by grounding — mostly invented defaults ("IV", "Tablet") and column shifts: the guard working as designed.
- **DrugBank-only classic dangers can be under-rated** (sertraline + tramadol etc.) → Batch 7.
- **Alert volume:** DrugBank records an interaction for ~half of all inpatient pairs; tiering, not the KB, controls what the reviewer sees.

---

## 2. Decision log

| Decision | Why | Where |
|---|---|---|
| LLM does extraction only; interaction facts only from KB | Core thesis — LLMs hallucinate / over-flag interactions | `athena/verification/` |
| Llama 3.2 3B Q4 (not 8B) | 8 GB unified memory with the rest of the stack | `configs/default.yaml` |
| Compact 8-column JSON rows | ~3× faster than named fields on the 3B model | `athena/extraction/llm_extractor.py` |
| Grounding guard drops any value not in the note | Anti-hallucination | `athena/extraction/align.py` |
| Rules ensemble + per-field precedence (chosen on **val** only) | Rules beat LLM on Dosage/Duration/Form | `athena/extraction/pipeline.py` |
| DDInter added beyond synopsis | Free DrugBank has no severity, no biologics | `athena/data/ddinter.py` |
| Wikidata + RxNorm for synonyms | Official DrugBank vocabulary download unavailable | `athena/normalize/lexicon.py` |
| Derived severity = majority vote over DDInter overlap per template | Data-driven, explainable; stored with evidence | `athena/verification/kb_build.py` |
| Route-specific DDInter concepts kept separate; salt merges by reviewed allow-list | Clinical correctness | `kb_build.py` |
| Review tier threshold 0.3 → 0.5 | 24 Review items on one real note | `configs/default.yaml` |
| Major never below Review | Safety override | `athena/fusion/report.py` |
| Audit log append-only + hash-chained | Accountability, tamper evidence | `athena/review/audit.py` |
| Demo uses synthetic notes only | Data-use agreements forbid showing n2c2/MIMIC text | `data/demo/` |
| No automatic learning from overrides | Overrides are patient-specific; self-modifying clinical software breaks traceability | §5 below |
| "Add prior medication list" re-runs the existing pipeline with the list attached as an admission section | Reconciliation without inventing backend features | `app/streamlit_app.py` |

### Declared deviations from the synopsis

| Synopsis says | Athena does | Why |
|---|---|---|
| DrugBank (DDI pairs) | DrugBank extract **+ DDInter** | Severity grades and biologics coverage |
| MIMIC-III | MIMIC-III **Demo** (open) | Notes come via n2c2 (built on MIMIC-III); demo `PRESCRIPTIONS` covers Branch B testing |
| n2c2 via official channel | Mirror copy for now | Official DBMI application in progress; files match official counts (303 / 202) |
| — | Wikidata + RxNorm for normalisation | See decision log |
| — | Synthetic training data (A + B) | n2c2 train is small; reconciliation edge cases rare |
| Kim, Kim & Choi used Llama 3 8B | Llama 3.2 3B | Memory budget |
| Fine-tuning phase | LoRA via MLX after the base demo | CPU-only fine-tuning impractical |
| Branch A = LLM extraction (scispaCy fallback) | LLM + rules ensemble; rules-only fallback | Ablation shows rules carry regular lists |
| XGBoost / LightGBM | XGBoost used for the **ML interaction predictor** (research component, not shown as verified) | Shtar et al. precedent |
| Fusion: extraction + DDI verification | + admission-vs-discharge reconciliation | Directly targets the reconciliation gap |

---

## 3. Open decision — Batch 7: high-alert combination layer

Some textbook-dangerous combinations exist only in the DrugBank extract under a generic template → derived Moderate → **Info** tier:

| Pair | Clinical risk | KB evidence today | Tier today |
|---|---|---|---|
| sertraline + tramadol | serotonin syndrome, seizures | DrugBank only, "neuroexcitatory activities" (11.5% Major) | Info |
| lisinopril + spironolactone | hyperkalaemia | DrugBank only, generic template (19.7% Major) | Info |
| oxycodone + lorazepam | respiratory depression (FDA boxed warning) | DrugBank only, generic template | Info |

| | Approach | Trade-off |
|---|---|---|
| **A (recommended)** | Small **cited** rule set of high-alert class combinations (≈10–15): opioid + benzodiazepine, serotonergic combinations, ACEi/ARB + K-sparing diuretic / K supplement, warfarin + NSAID/antiplatelet, QT-prolonging combinations. Class membership from RxNorm/ATC; each rule stores its citation; severity **Major**, basis `rule`. | Deterministic, auditable; declared addition; citations must be verified against primary sources |
| B | Floor: any DrugBank-recorded interaction ≥ Review | Alert fatigue returns (~16 Moderate items per patient) |
| C | Document only | Demo visibly under-rates classic dangers |

**Done when:** the three pairs reach Critical/Review with a cited rule in the explanation; tests pass; Review volume re-measured. Pinned today by `test_known_limitation_opioid_benzo_not_major`.

---

## 4. Future work

### Phase 2 — Improve (8 Oct – 31 Oct)

| Batch | Deliverable | Done when |
|---|---|---|
| **7 — High-alert combinations** | See §3 | Decision A/B/C taken and implemented |
| **8 — Synthetic data A** | Template generator + validator; edge-case suite (abbreviations, brands, held/stopped, dose changes) | Labels/offsets validated by code |
| **9 — Synthetic data B** | Claude-written notes from MIMIC-demo med lists, validated and versioned in `data/synthetic/` | Discarded if labels don't align |
| **10 — Fine-tuning** | LoRA fine-tune of the 3B model (MLX) on n2c2 train + synthetic → GGUF → Ollama (`athena-extract:3b`) | Compared against LLM-only baseline (.659) and hybrid (.794) |
| **11 — Similarity model (full)** | Extend the XGBoost predictor: cold-start split, SMILES/structure features (RDKit), 86-class mechanism prediction, LightGBM comparison; surface in the UI only as *predicted / unverified* for pairs the KB doesn't cover | Cold-start results reported; UI label reviewed |
| **12 — FAERS signal** | Co-report statistics as an extra, clearly labelled evidence feature | Never a sole basis for a flag |
| **13 — Fusion tuning + route awareness** | Calibrate weights/thresholds on synthetic planted-interaction patients; use extracted Route for route-specific concepts (lidocaine patch) | Sensitivity for Major, false alarms per patient reported |

### Phase 3 — Evaluate & write (1 Nov – 24 Nov, buffer to mid-Dec)

| Batch | Deliverable |
|---|---|
| **14 — Full evaluation** | Full 202-note n2c2 test; all metrics; ablations (LLM vs rules vs hybrid, ± synthetic, ± fine-tuning, Llama 3.2 3B vs Phi-3-mini); resource measurements; normaliser accuracy on a hand-checked sample |
| **15 — Report** | Final report, limitations, figures |
| **16 — Paper** | Short paper draft (if results justify it) |

### Ideas / refinements backlog
- Accept a **separate prior medication list** as a first-class input (today: attached to the note text by the UI).
- Capture **corrected values** on "extraction error" overrides so they become labelled training data.
- Reviewer role stored in the audit log (needs a schema version bump).
- Real authentication (hospital SSO) instead of a local session name.
- Phi-3-mini comparison model run.
- Derived-severity accuracy (leave-one-out against DDInter) — confusion matrix.

---

## 5. Governed feedback loop (design, not implemented)

Athena does **not** learn automatically from overrides (patient-specific; would break traceability and reproducibility; regulators expect controlled change). The audit log supports an **offline, human-governed** loop:

```
audit log → periodic CSV export → override-pattern analysis (by pair, template, tier, reason)
→ expert review → versioned change (fusion weights, rule, fine-tuning data)
→ re-run n2c2 + synthetic evaluation → release new version (version recorded in every report)
```

| Override reason | Could improve | How |
|---|---|---|
| Extraction error | Branch A | Labelled correction data for fine-tuning (needs corrected value captured) |
| Not clinically relevant (repeated) | Fusion thresholds | Recalibration evidence; **Major never auto-suppressed** |
| Already managed | — | Patient context, not knowledge |
| Confirms | Calibration | Tier precision |

---

## 6. Synthetic data rules (Batches 8–9)

- Synthetic data is **never** the final test set; headline results are on the real n2c2 test set.
- **No n2c2 text is ever sent to a cloud model** (data-use agreement). Generator B is seeded only from open MIMIC-demo data.
- Results reported **with and without** synthetic data; real vs synthetic scores separately.
- Synthetic data is **never** used to create interaction facts.
- Use of a cloud model for dev-time data generation is declared in the report.

---

## 7. To-do (non-batch)

- [ ] Full 202-note n2c2 test run (≈ 3–4 h LLM time, overnight): `python scripts/warm_llm_cache.py test` then `python scripts/eval_extraction.py test --save`
- [ ] Official n2c2 access via the DBMI portal (then re-verify files against the mirror)
- [ ] Decide GitHub repo visibility (currently **public**; README mentions the n2c2 mirror) — consider private until DBMI access
- [ ] DrugBank academic licence / vocabulary download when available (swap in official 5.1.x)
- [ ] Batch 7 decision (A/B/C)

---

## 8. Demo-day checklist

```bash
cd ~/Desktop/minor
brew services start ollama          # local LLM (stop afterwards: brew services stop ollama)
source .venv/bin/activate
streamlit run app/streamlit_app.py  # http://localhost:8501
```

- [ ] Pre-warm hybrid mode on the 3 demo notes (first hybrid run ≈ 30–60 s, cached afterwards)
- [ ] Fallback: sidebar → **Rules only** (instant)
- [ ] Shortcut: `http://localhost:8501/?demo=1&reviewer=Your%20Name` (demo=1..3)
- [ ] Practice clicks are permanent in the audit log — use a test reviewer name, or set `ATHENA_AUDIT_DB` to a scratch file
- [ ] Results to show: README §Results, `docs/results/ddi_xgboost_confusion.png`, live `python scripts/eval_extraction.py test --limit 40` (~1 min, cached)

**Suggested flow (≈5 min):** About page → demo patient 1 (warfarin Critical findings, duplicate acetaminophen, Admission → Discharge) → *View evidence & reasoning* → Confirm one, Override one → Audit log ("Chain intact") → results slide.

**Weaknesses to raise proactively:** under-rated classic pairs (Batch 7); rules ≈ LLM on regular lists; test on 35 / 202 notes; route-unaware topical drugs; no automatic learning (by design); XGBoost predictor is warm-start only and not shown as verified.

---

## 9. Environment notes

| Constraint | How it is met |
|---|---|
| 8 GB unified memory, no GPU (Apple M2) | 3B model at 4-bit (~2.3 GB runtime); KB in SQLite; FAERS processed in chunks |
| Local only at inference | Ollama on `localhost`; Streamlit bound to localhost, telemetry off; fonts served locally |
| Python 3.10+ | Python **3.11** venv (system Python 3.14 unsupported by spaCy 3.7) |

- numpy pinned < 2 (spaCy 3.7 / thinc 8.2 wheels); macOS needs `brew install libomp` for XGBoost.
- Disk: ~5–7 GB total; +6–8 GB temporarily during fine-tuning.
- `ATHENA_AUDIT_DB=/path/scratch.db` redirects the audit log (demos, automated tests).
