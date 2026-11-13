# Model Comparison: Clinical Extraction on n2c2 2018 (202 Test Notes)

Evaluation of local Ollama models on Branch A (202 test notes, zero-shot `extract_v1` prompt, temperature 0, seed 42).
All models use identical chunking (4,000 chars, 300 char overlap) and grounding guard rules.

| Model | Hybrid Strict F1 (95% CI) | Hybrid Lenient F1 (95% CI) | LLM Strict F1 (95% CI) | LLM Lenient F1 (95% CI) | Avg Sec / Note | Peak RAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **llama3.2:3b (baseline)** | 0.8171 [0.8063, 0.8273] | 0.8800 [0.8719, 0.8878] | 0.6662 [0.6494, 0.6821] | 0.7341 [0.7176, 0.7491] | 71.3s | 3.8 GB |
| **gemma2:2b** | 0.7825 [0.7691, 0.7951] | 0.8524 [0.8414, 0.8627] | 0.5901 [0.5724, 0.6069] | 0.6712 [0.6530, 0.6879] | 42.8s | 2.8 GB |
| **qwen2.5:3b** | 0.8110 [0.8009, 0.8210] | 0.8772 [0.8697, 0.8845] | 0.5937 [0.5758, 0.6113] | 0.6641 [0.6464, 0.6823] | 28.4s | 3.2 GB |
| **phi3.5:3.8b** | 0.7952 [0.7841, 0.8055] | 0.8626 [0.8544, 0.8702] | 0.5969 [0.5804, 0.6137] | 0.6663 [0.6504, 0.6828] | 55.6s | 3.6 GB |
| **qwen2.5:7b** | 0.8195 [0.8089, 0.8297] | 0.8871 [0.8809, 0.8931] | 0.6344 [0.6127, 0.6549] | 0.7005 [0.6779, 0.7207] | 73.8s | 5.5 GB |
| **llama3.1:8b** | 0.8298 [0.8207, 0.8385] | 0.8894 [0.8827, 0.8958] | 0.7450 [0.7305, 0.7586] | 0.8043 [0.7903, 0.8166] | 113.2s | 5.8 GB |

### Notes and Methodology
- **Hybrid**: Ensemble combining scispaCy NER, deterministic regex/dictionary rules, and LLM structured extraction, adjudicated by the grounding guard.
- **LLM-only**: Direct structured output from the LLM after grounding guard filter (invented attributes dropped).
- **Confidence intervals**: 95% bootstrap confidence intervals computed over 2,000 resamples across clinical notes.
- **Hardware**: Apple Silicon M2, unified memory, Metal acceleration.
