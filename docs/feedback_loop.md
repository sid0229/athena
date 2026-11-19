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

## 4. Empirical Evaluation & Results

### 4.1 Benchmark Evaluation Setup
- Evaluated on n2c2 2018 Track 2 clinical discharge summaries.
- Tested on standard test benchmark notes with local `llama3.2:3b` via Ollama.
- Metrics evaluated:
  - Strict and Lenient micro F1 across 7 medication slots.
  - 2,000 paired bootstrap iterations over notes for 95% confidence intervals.
  - Alert-level evaluation against gold annotations (InteractionChecker & DDInter).

### 4.2 Extraction Benchmark (Hybrid Mode)

| Entity / Metric | Baseline Hybrid | Hybrid + Dynamic Few-Shot | Difference |
|---|---|---|---|
| **Drug (Strict F1)** | **0.860** | 0.855 | -0.005 |
| **Strength (Strict F1)** | 0.818 | **0.851** | **+0.033** |
| **Route (Strict F1)** | 0.874 | **0.885** | **+0.011** |
| **Frequency (Strict F1)** | 0.712 | **0.738** | **+0.026** |
| **Form (Strict F1)** | 0.764 | **0.779** | **+0.015** |
| **Dosage (Strict F1)** | **0.698** | 0.697 | -0.001 |
| **Duration (Strict F1)** | **0.485** | 0.404 | -0.081 |
| **Strict Micro F1** | 0.8054 [0.7772, 0.8280] | **0.8151** [0.7924, 0.8348] | **+0.0097** (95% CI [-0.0043, +0.0248]) |
| **Lenient Micro F1** | 0.8748 [0.8584, 0.8892] | **0.8794** [0.8624, 0.8948] | **+0.0046** (95% CI [-0.0045, +0.0143]) |
| **LLM Values Dropped (Ungrounded)** | 47.8% (1,349 / 2,822) | **14.8%** (274 / 1,856) | **-33.0% (3.2x reduction)** |

### 4.3 Alert-Level Evaluation

| Metric | Baseline System | System + Dynamic Few-Shot | Delta |
|---|---|---|---|
| **Total Gold Pairs** | 992 | 992 | — |
| **Total System Pairs** | 965 | 984 | +19 |
| **True Positive Pairs** | 963 | 977 | +14 |
| **Pair Precision** | **0.9979** | 0.9929 | -0.0050 |
| **Pair Recall** | 0.9708 | **0.9849** | **+0.0141 (+1.4%)** |
| **Gold Major Pairs** | 37 | 37 | — |
| **Major Pairs Found** | 37 | 37 | 0 (100% recall) |
| **Major Recall** | **1.0000** | **1.0000** | **0.0000 (No drop)** |
| **Major Reaching Actionable Tier** | **1.0000** | **1.0000** | **0.0000 (No drop)** |
| **Actionable Alert Precision** | 0.9948 | **0.9950** | **+0.0002** |
| **Actionable Alerts / Note** | 5.46 | 5.77 | +0.31 |

---

## 5. Observations & Promotion Decision

### 5.1 Positive Findings
1. **Dramatic Grounding Improvement**: Unmatched/hallucinated LLM attribute values dropped from 47.8% to 14.8% (a 3.2x reduction). The few-shot examples clearly anchor the LLM to verbatim spans rather than expanded clinical synonyms.
2. **Attribute Accuracy Gains**: Substantial improvements in Strength (+0.033), Frequency (+0.026), and Route (+0.011) F1.
3. **Alert Recall Gain**: Overall DDI pair recall increased by +1.4% (0.9708 &rarr; 0.9849), while Major recall was perfectly preserved at 100%.

### 5.2 Negative Result & Critical Trade-Off
1. **Drug Entity Recall Slight Drop**: Drug entity strict F1 slightly dropped from 0.860 to 0.855 (-0.005). The few-shot prompt causes the smaller 3B model to become slightly conservative on unusual narrative drug mentions outside standard list formatting.
2. **Inference Latency Overhead**: Few-shot context adds ~400-600 prompt tokens per chunk, increasing generation latency from ~1.5s to ~2.8s per chunk.

### 5.3 Promotion Gate Decision
- **Gate Rule**: Only enable `extraction.use_corrections` by default if overall F1 does not drop AND Major recall does not drop.
- **Outcome**: While overall strict micro F1 increased (+0.0097) and Major recall held at 100%, Drug entity F1 experienced a small regression (-0.005) and LLM call latency increased ~1.8x.
- **Decision**: In accordance with clinical safety principles where drug omission is higher risk than attribute variance, **Level 1 learning is kept opt-in (`extraction.use_corrections: false` by default)**. Users and clinical sites can enable it via `configs/default.yaml` or programmatically when attribute precision and grounding are prioritized.

---

## 6. Level 2: Supervised Instruction Fine-Tuning (LoRA / MLX)

### 6.1 Exploration Overview
- Target Model: `llama3.2:3b`.
- Dataset: High-confidence reviewer corrections merged with the n2c2 training gold standard formatted in the Athena 8-column JSON schema.
- Training Paradigm: QLoRA (4-bit quantization, rank $r=16$, $\alpha=32$) using Apple MLX (`mlx-lm`) optimized for unified memory on Apple Silicon.
- Deployment Path: Convert trained LoRA adapters to GGUF format for zero-overhead local execution in Ollama.
- Gate Requirement: Model adapter is promoted to production only if test micro F1 and Major interaction recall do not regress compared to the base baseline.
