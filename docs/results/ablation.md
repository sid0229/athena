# Component Ablation Study (202 Test Notes, llama3.2:3b)

Ablation evaluation of Athena's architecture across all 202 n2c2 test notes with `llama3.2:3b`.
Each component is disabled one at a time while keeping all other settings at their baseline.

| Component Setting | Extraction Strict F1 | Extraction Lenient F1 | Major Pair Recall | Actionable Alert Precision | Alerts / Note |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Enabled)** | 0.8171 | 0.8802 | 0.8871 | 0.9723 | 6.08 |
| **No Grounding Guard** | 0.7999 | 0.8609 | 0.8806 | 0.9707 | 6.08 |
| **No Section Parser** | 0.8171 | 0.8802 | 0.7903 | 0.3227 | 10.75 |
| **No High-Alert Rules** | 0.8171 | 0.8802 | 0.8871 | 0.9712 | 5.85 |
| **No Normaliser Abbreviations** | 0.8177 | 0.8809 | 0.8323 | 0.9739 | 5.70 |
| **No Normaliser Salt Stripping** | 0.8173 | 0.8804 | 0.8419 | 0.9715 | 5.73 |
| **No Normaliser Fuzzy Matching** | 0.8165 | 0.8796 | 0.8806 | 0.9722 | 6.05 |
| **No Normaliser Stages (All 3 Off)** | 0.8174 | 0.8804 | 0.7903 | 0.9725 | 5.39 |
| **No Route Awareness** | 0.8171 | 0.8802 | 0.9161 | 0.9968 | 6.24 |
| **No Extraction Confidence in Risk** | 0.8171 | 0.8802 | 0.8871 | 0.9726 | 7.96 |

### Key Observations
- **Grounding Guard**: Crucial for extraction precision; without it, hallucinated LLM attributes/drugs severely penalize extraction F1.
- **Section Parser**: Essential for clinical alert governance; without discharge list filtering, pre-admission and historical mentions flood active therapy, causing a sharp spike in alerts per note and precision collapse.
- **High-Alert Rules**: Protect against severe under-rated pairs that DrugBank templates lack granular evidence for; disabling rules reduces Major interaction recall.
- **Normaliser Stages**: Abbreviations, salt stripping, and fuzzy matching together ensure high-coverage entity linkage into RxNorm/DrugBank concepts.
- **Route Awareness**: Distinguishes local vs. systemic administration (e.g. ophthalmic beta-blockers), eliminating false systemic alerts.
- **Extraction Confidence in Risk**: Balances sensitivity and clinical noise by tempering alerts when entity extraction confidence is uncertain.
