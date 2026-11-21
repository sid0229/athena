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

## 6. Level 2: Supervised Instruction Fine-Tuning (LoRA / MLX) (`exp/lora-eval`)

### 6.1 Exploration & Architecture
- **Target Model**: `llama3.2:3b-instruct` (4-bit base via `mlx-community/Llama-3.2-3B-Instruct-4bit`).
- **Data Preparation & Leakage Separation (`scripts/prepare_lora_data.py`)**:
  - **Training split**: 5,399 clinical chunk examples combining n2c2 2018 Track 2 train gold notes and train-split snippet corrections formatted as JSONL chat messages.
  - **Validation split**: 134 clinical chunks from n2c2 validation gold notes.
  - **Test split**: 0 test examples used during preparation or training (strictly zero leakage). Out-of-sample evaluation was performed entirely on the held-out 202 test notes (769 chunks).
  - Formatted strictly to the 8-column schema: `[drug, strength, dosage, form, route, frequency, duration, status]`.
- **Training Paradigm (`scripts/train_lora.py`)**:
  - Low-Rank Adaptation (QLoRA) utilizing Apple MLX (`mlx-lm`) on Apple Silicon unified memory.
  - Hyperparameters: Rank $r=16$, Scaling $\alpha=32$, Dropout $0.0$, 16 LoRA attention projection layers, batch size $2$, AdamW with learning rate $1\times 10^{-4}$.
  - Training performance: Loss dropped from 2.428 to 1.365 over iterations with validation loss improving from 2.400 to 1.920, training at ~578 tokens/sec on Apple M-series GPU.

### 6.2 Uncompressed Fusion, GGUF Conversion, & Ollama Serving
1. **Resolving the Ollama Dequantization Blocker**:
   - Initial attempts to register 4-bit MLX safetensors directly with Ollama failed (`panic: mlx: [dequantize] Invalid quantization mode ''`).
   - The adapter was re-fused with full dequantization into uncompressed 16-bit safetensors at `data/processed/lora/fused_fp16` (6.4 GB):
     ```bash
     .venv/bin/python3 -m mlx_lm fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
       --adapter-path data/processed/lora/adapters \
       --save-path data/processed/lora/fused_fp16 --dequantize
     ```
2. **GGUF Conversion & Quantization**:
   - Converted the uncompressed weights to standard GGUF format via `convert_hf_to_gguf.py` (`data/processed/lora/athena-lora-f16.gguf`).
   - Quantized the GGUF model to 4-bit `Q4_K_M` using `llama-quantize` (`data/processed/lora/athena-lora-q4_k_m.gguf`, 1.9 GB):
     ```bash
     llama-quantize data/processed/lora/athena-lora-f16.gguf data/processed/lora/athena-lora-q4_k_m.gguf Q4_K_M
     ```
   - Created the Ollama model `athena-lora:3b` with the standard Llama 3.2 chat template.
3. **Constrained Grammar Decoding**:
   - In MLX unconstrained generation, 3B models tend to produce synthetic list continuations (hallucinating extra drugs past the chunk boundary).
   - Under Ollama, Athena passes the strict 8-string JSON schema (`format: _SCHEMA`), guaranteeing valid JSON row syntax and preventing run-away hallucinations.

### 6.3 Test Set Evaluation (202 Notes)

#### Extraction Performance
Scored on all 202 held-out test notes (769 chunks) using `scripts/eval_extraction.py test --modes llm,hybrid --save`:

| Configuration | Strict Micro F1 (95% CI) | Lenient Micro F1 (95% CI) | Ungrounded Attr Discard | Sec / Note |
|---|:---:|:---:|:---:|:---:|
| **LLM-only (`athena-lora:3b`)** | 0.5555 [0.5278, 0.5807] | 0.6117 [0.5819, 0.6380] | **15.3%** (1,320 / 8,600) | — |
| **Baseline Hybrid (`llama3.2:3b`)** | 0.8171 [0.8064, 0.8274] | 0.8802 [0.8721, 0.8880] | 43.0% (7,953 / 18,481) | 71.3 s |
| **Fine-Tuned Hybrid (`athena-lora:3b`)** | **0.8174** [0.8071, 0.8269] | **0.8831** [0.8758, 0.8901] | **15.3%** (1,320 / 8,600) | **25.2 s** |

**Paired Bootstrap Differences (2,000 resamples: `athena-lora:3b` vs `llama3.2:3b`)**:
- **Strict Micro F1**: $+0.0003$ ($95\%\text{ CI } [-0.0068, +0.0072]$) — statistically indistinguishable from baseline.
- **Lenient Micro F1**: $+0.0029$ ($95\%\text{ CI } [-0.0015, +0.0079]$).
- **Grounding Discard Rate**: Ungrounded LLM attribute spans dropped dramatically from **43.0% down to 15.3%** (a 2.8x hallucination reduction).
- **Inference Speed**: Total cache generation for 202 notes was 5,085.6 seconds (**25.2 s / note**), representing a **2.8x speedup** over baseline `llama3.2:3b` (71.3 s / note) due to compact, non-redundant token generation.

#### Alert-Level Evaluation
Scored through the full clinical pipeline (discharge sectioning, status inference, DDInter/DrugBank verification, high-alert rules, and fusion) on the 202 test notes:

| Metric | Baseline (`llama3.2:3b`) | Fine-Tuned (`athena-lora:3b`) | Difference |
|---|:---:|:---:|:---:|
| **Major Alert Recall** | **91.59%** (414 / 452) | **91.15%** (412 / 452) | **-0.44%** (-2 pairs) |
| **Major Reaching Actionable Tier** | 91.59% (414 / 452) | 91.15% (412 / 452) | -0.44% |
| **Pair Precision** | 98.74% (7,515 / 7,611) | 98.11% (7,389 / 7,531) | -0.63% |
| **Pair Recall** | 92.06% (7,515 / 8,163) | 90.52% (7,389 / 8,163) | -1.54% |
| **Actionable Alert Precision** | 99.55% (1,986 / 1,995) | 99.16% (1,885 / 1,901) | -0.39% |
| **Actionable Alerts / Note** | 9.88 | 9.41 | -0.47 |

### 6.4 Promotion Gate Decision
- **Gate Rule**: Promote to default model only if hybrid test F1 does not drop ($\ge 0.8171$) AND Major alert recall does not drop ($\ge 91.59\%$).
- **Evaluation Finding**:
  - Hybrid strict F1 met the threshold ($0.8174 \ge 0.8171$, paired difference $+0.0003$, $95\%\text{ CI } [-0.0068, +0.0072]$).
  - Major alert recall regressed slightly by $-0.44\%$ ($91.15\% < 91.59\%$, finding 412 instead of 414 Major interactions).
- **Decision**: **FAILS the strict promotion gate for default adoption.**
- **Architectural Conclusion**:
  - `llama3.2:3b` remains the default production model in `configs/default.yaml`.
  - `athena-lora:3b` is preserved and documented as an optional model configuration (`llm.model: athena-lora:3b`). It provides substantial operational benefits for high-throughput or resource-constrained deployments (2.8x faster inference at 25.2s/note and 2.8x fewer ungrounded attribute hallucinations) while maintaining near-parity on safety-critical alerts (91.15% vs 91.59% Major recall).


