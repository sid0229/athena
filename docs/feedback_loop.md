# Reviewer Feedback Loop & Continual Learning in Athena

This document details the design, implementation, and empirical evaluation of Athena's reviewer feedback loop across three learning tiers:
1. **Level 0 (Online Reviewer Corrections)**: Interactive UI corrections and deterministic re-analysis without re-extraction.
2. **Level 1 (Retrieval-Augmented Dynamic Few-Shot Learning)**: Prompt augmentation with the $k=2..3$ most similar stored corrections, zero model retraining, with cache signature invalidation.
3. **Level 2 (Parametric Fine-Tuning / LoRA)**: Supervised instruction fine-tuning via QLoRA.

---

## 1. System Architecture

```
+-------------------------------------------------------------------------------+
|                             REVIEWER WORKFLOW                                 |
+-------------------------------------------------------------------------------+
                                      |
                       [Medications Page / Corrections UI]
                                      |
                      Add missed drug / remove wrong drug /
                      edit strength, route, frequency, status
                                      |
                             POST /api/correct
                                      |
          +---------------------------+---------------------------+
          |                                                       |
          v                                                       v
  [data/corrections.db]                                   [Audit System]
  (SQLite, gitignored)                                (data/processed/audit.db)
  - note_hash, note_text                              - action: "correct"
  - original_extraction                               - prev_hash -> current_hash
  - corrected_list                                    - cryptographic hash chain
  - reviewer, timestamp, case_id
          |
          +---------------------------+
                                      |
                                      v
                      [Level 1: Dynamic Few-Shot Bank]
                      - TF-IDF / rapidfuzz retriever over chunks
                      - 2-3 most similar verified examples
                      - Injected into chat prompt (zero retraining)
                      - Cache signature: includes bank version
                      - Config: extraction.use_corrections: false
```

---

## 2. Level 0: Online Reviewer Corrections (`feat/med-corrections`)

### 2.1 UI & Endpoint Design
- On the **Medications** page (`web/src/pages/Medications.tsx`), reviewers can toggle edit mode to:
  - Add missed drugs with dosage, route, and frequency.
  - Remove spurious extractions.
  - Modify attributes (strength, route, frequency, status) inline.
- When saved, the frontend calls `POST /api/correct`:
  - **Bypasses LLM extraction entirely**: calls `analyze_medications` directly.
  - Executes normalization (`DrugNormalizer`), interaction checking (`InteractionChecker`), and multi-source evidence fusion (`fuse_findings`).
  - Returns updated findings, active alerts, and a refreshed report hash.

### 2.2 Local Storage & Audit Chaining
- **SQLite Storage (`data/corrections.db`)**:
  - Schema: `corrections(id, ts, note_hash, original_extraction, corrected_list, reviewer, case_id, note_text)`.
  - Local, gitignored, persistent across app restarts.
- **Audit Logging**:
  - Added new action type `"correct"` to `athena.review.audit.ACTIONS`.
  - Stored in `data/processed/audit.db` with SHA-256 hash chaining (`prev_hash` &rarr; `current_hash`).
  - Chain integrity verification (`audit.verify()`) remains unbroken after corrections.

---

## 3. Level 1: Retrieval-Augmented Dynamic Few-Shot Learning (`feat/correction-fewshot`)

### 3.1 Retrieval & Prompt Assembly
- Implemented in `athena/extraction/corrections_fewshot.py`:
  - `CorrectionRetriever`: Computes similarity between incoming clinical text chunks and verified examples using sublinear TF-IDF (word 1-2 grams) or `rapidfuzz` token-set ratio.
  - Retrieves $k=2$ most relevant historical corrections.
  - Injected as worked assistant turns before the user chunk query:
    ```json
    {"role": "user", "content": "<similar verified clinical snippet>"}
    {"role": "assistant", "content": "{\"meds\": [[\"metoprolol tartrate\", \"25 mg\", ...]]}"}
    {"role": "user", "content": "<current clinical note chunk>"}
    ```
- **Zero Retraining**: Pure in-context learning.

### 3.2 Cache Signature & Versioning
- `OllamaExtractor` cache key is strictly versioned:
  - Cache signature includes `{"use_corrections": true, "correction_bank_version": "v_<sha256>"}`.
  - Outputs stored under dedicated cache directory: `data/processed/llm_cache/llama3.2_3b_extract_v1_corr_<bank_version>/`.
  - Zero collision with baseline cache (`llama3.2_3b_extract_v1`), guaranteeing reproducible benchmarks.

### 3.3 Leakage Prevention
- For benchmarking, the correction bank is built strictly from the `train` split of n2c2 (`split="train"`), containing 4,506 verified snippet-row pairs.
- **Zero test notes are ever present in the retrieval bank**, eliminating data contamination.

---

## 4. Empirical Evaluation Across All 202 Test Notes

### 4.1 Benchmark Evaluation Setup
- Evaluated on all 202 n2c2 2018 Track 2 clinical discharge summaries (769 chunks).
- Full caching and evaluation executed with local `llama3.2:3b` via Ollama.
- Metrics evaluated:
  - Strict and Lenient micro F1 across all 7 medication slots.
  - 2,000 paired bootstrap iterations over notes for 95% confidence intervals.
  - Alert-level evaluation against gold annotations (InteractionChecker & DDInter).

### 4.2 Extraction Benchmark (202 Test Notes, Hybrid Mode)

| Entity / Metric | Baseline Hybrid | Hybrid + Dynamic Few-Shot | Difference |
|---|---|---|---|
| **Drug (Strict F1)** | **0.870** (P 0.883 / R 0.856) | 0.856 (P 0.862 / R 0.850) | -0.014 |
| **Strength (Strict F1)** | **0.844** (P 0.837 / R 0.851) | 0.839 (P 0.834 / R 0.843) | -0.005 |
| **Route (Strict F1)** | **0.893** (P 0.934 / R 0.855) | 0.891 (P 0.939 / R 0.847) | -0.002 |
| **Frequency (Strict F1)** | 0.670 (P 0.656 / R 0.684) | **0.699** (P 0.705 / R 0.693) | **+0.029** |
| **Form (Strict F1)** | 0.815 (P 0.828 / R 0.802) | **0.818** (P 0.840 / R 0.797) | **+0.003** |
| **Dosage (Strict F1)** | **0.718** (P 0.970 / R 0.570) | 0.715 (P 0.967 / R 0.567) | -0.003 |
| **Duration (Strict F1)** | **0.495** (P 0.745 / R 0.370) | 0.472 (P 0.740 / R 0.347) | -0.023 |
| **Strict Micro F1** | **0.8171** [0.8064, 0.8274] | 0.8158 [0.8028, 0.8279] | **-0.0013** (95% CI [-0.0107, +0.0073]) |
| **Lenient Micro F1** | **0.8802** [0.8721, 0.8880] | 0.8793 [0.8700, 0.8878] | **-0.0009** (95% CI [-0.0082, +0.0059]) |
| **LLM Values Dropped (Ungrounded)** | 43.0% (7,953 / 18,481) | **17.4%** (2,160 / 12,415) | **-25.6% (2.5x grounding improvement)** |

### 4.3 Alert-Level Evaluation (202 Test Notes)

| Metric | Baseline System | System + Dynamic Few-Shot | Delta |
|---|---|---|---|
| **Total Gold Pairs** | 8,163 | 8,163 | — |
| **Total System Pairs** | 7,611 | 7,432 | -179 |
| **True Positive Pairs** | 7,515 | 7,346 | -169 |
| **Pair Precision** | 0.9874 | **0.9884** | **+0.0010** |
| **Pair Recall** | **0.9206** | 0.8999 | -0.0207 (-2.1%) |
| **Gold Major Pairs** | 452 | 452 | — |
| **Major Pairs Found** | **414** | 409 | -5 |
| **Major Recall** | **0.9159** (91.6%) | 0.9049 (90.5%) | **-0.0110 (-1.1%)** |
| **Major Reaching Actionable Tier** | **0.9159** (91.6%) | 0.9049 (90.5%) | **-0.0110 (-1.1%)** |
| **Actionable Alert Precision** | 0.9955 | **0.9970** | **+0.0015** |
| **Actionable Alerts / Note** | 9.88 | 9.76 | -0.12 |

---

## 5. Negative Results Analysis & Promotion Gate Decision

### 5.1 Analysis of the Negative Result
1. **Drug Entity Omission in Narrative Contexts**:
   - Drug strict F1 fell from 0.870 to 0.856 (-0.014), and Drug recall dropped from 0.856 to 0.850.
   - When primed with structured few-shot examples, `llama3.2:3b` develops a strong inductive bias towards tabular/list-formatted discharge prescriptions. As a consequence, it occasionally skips informal or implicit drug mentions buried deep within prose (e.g., *"patient was placed on an amiodarone drip in the ICU"*).
2. **Cascading Impact on Downstream Alert Recall**:
   - Because interaction detection depends entirely on identifying both drugs in a pair, the slight drop in Drug recall directly translated to 5 fewer Major interaction pairs detected (414 &rarr; 409), reducing Major recall from 91.59% to 90.49% (-1.10%).
3. **Inference Latency Trade-Off**:
   - Incorporating 2 dynamic few-shot examples increased average chunk generation time from ~1.5s to ~2.8s on Apple M2 (an 86% latency penalty).

### 5.2 Positive Offsetting Strengths
- **Grounding Fidelity**: LLM attribute values discarded by grounding dropped dramatically from 43.0% down to 17.4% (a 2.5x hallucination reduction).
- **Attribute Precision**: Frequency strict F1 increased noticeably (+0.029, from 0.670 to 0.699).
- **Precision on Actionable Alerts**: Actionable alert precision reached 99.70% (1,965 / 1,971).

### 5.3 Formal Promotion Decision
- **Gate Rule**: Only enable `extraction.use_corrections` by default if overall F1 does not drop AND Major recall does not drop.
- **Evaluation Finding**:
  - Strict micro F1 changed by $-0.0013$ ($95\%\text{ CI } [-0.0107, +0.0073]$).
  - Major recall dropped by $-0.0110$ ($91.59\% \rightarrow 90.49\%$).
- **Decision**: **FAILS promotion gate for default adoption.**
  Level 1 learning is kept **strictly opt-in (`extraction.use_corrections: false` by default in `configs/default.yaml`)**. It remains fully accessible for sites that prioritize verbatim grounding and attribute fidelity over maximal prose-drug recall.

---

## 6. Level 2: Supervised Instruction Fine-Tuning (LoRA / MLX)

### 6.1 Exploration Overview
- Target Model: `llama3.2:3b`.
- Dataset: High-confidence reviewer corrections merged with the n2c2 training gold standard formatted in the Athena 8-column JSON schema.
- Training Paradigm: QLoRA (4-bit quantization, rank $r=16$, $\alpha=32$) using Apple MLX (`mlx-lm`) optimized for unified memory on Apple Silicon.
- Deployment Path: Convert trained LoRA adapters to GGUF format for zero-overhead local execution in Ollama.
- Gate Requirement: Model adapter is promoted to production only if test micro F1 and Major interaction recall do not regress compared to the base baseline.
