# Component Ablation Study (202 Test Notes, llama3.2:3b)

Ablation evaluation of Athena's architecture across all 202 n2c2 test notes with `llama3.2:3b`.
Each component is disabled one at a time while keeping all other settings at their baseline.

| Component Setting | Extraction Strict F1 | Extraction Lenient F1 | Major Pair Recall | Major Actionable Rate | Actionable Alert Precision | Alerts / Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (All Enabled)** | 0.8171 | 0.8802 | 0.9159 | 0.9159 | 0.9955 | 9.88 |
| **No Grounding Guard** | 0.7999 | 0.8609 | 0.9204 | 0.9204 | 0.9975 | 9.87 |
| **No Section Parser** | 0.8171 | 0.8802 | 0.8142 | 0.8142 | 0.3554 | 17.40 |
| **No High-Alert Rules** | 0.8171 | 0.8802 | 0.9159 | 0.9159 | 0.9955 | 9.88 |
| **No Normaliser Abbreviations** | 0.8177 | 0.8809 | 0.8783 | 0.8783 | 0.9969 | 9.47 |
| **No Normaliser Salt Stripping** | 0.8173 | 0.8804 | 0.8230 | 0.8230 | 0.9951 | 9.07 |
| **No Normaliser Fuzzy Matching** | 0.8165 | 0.8796 | 0.9115 | 0.9115 | 0.9955 | 9.85 |
| **No Normaliser Stages (All 3 Off)** | 0.8174 | 0.8804 | 0.7876 | 0.7876 | 0.9960 | 8.68 |
| **No Route Awareness** | 0.8171 | 0.8802 | 0.8982 | 0.8982 | 0.9630 | 10.02 |
| **No Extraction Confidence in Risk** | 0.8171 | 0.8802 | 0.9159 | 0.9159 | 0.9926 | 12.08 |

### Key Observations
- **Grounding Guard**: Critical for extraction reliability; removing the guard lowers strict F1 from 0.8171 to 0.7999 (-0.0172) and lenient F1 from 0.8802 to 0.8609 (-0.0193) due to ungrounded LLM hallucinations.
- **Section Parser**: Essential for clinical alert governance; without discharge list filtering, pre-admission and historical mentions flood active therapy, causing a sharp surge in alerts per note (9.88 to 17.40) and severe precision collapse (0.9955 to 0.3554).
- **High-Alert Rules**: In this test cohort, high-alert rules elevate severe interactions (e.g. QT-prolonging fluoroquinolone + macrolide pairs) to the top-priority Critical tier (risk >= 0.85). Because their baseline DrugBank evidence already scores sufficiently high to reach the Review tier (risk >= 0.45), pair recall and overall actionable rate remain stable (0.9159), while the rules guarantee prompt Critical triage.
- **Normaliser Stages**: Disabling all three normaliser stages causes Major pair recall to plunge from 0.9159 to 0.7876 (-12.8 points) and alerts per note to drop from 9.88 to 8.68 as unmapped variants fail KB lookup. Individually, salt stripping is the most critical stage (recall drops to 0.8230), followed by abbreviations (recall drops to 0.8783) and fuzzy matching (recall drops to 0.9115).
- **Route Awareness**: Distinguishes local vs. systemic administration (e.g. ophthalmic beta-blockers and topical NSAIDs). Disabling route awareness causes actionable alert precision to fall from 0.9955 to 0.9630, adds spurious systemic alerts (alerts per note rise from 9.88 to 10.02), and lowers Major recall against route-specific gold from 0.9159 to 0.8982.
- **Extraction Confidence in Risk**: Balances sensitivity and clinical noise by tempering alert thresholds when entity extraction confidence is uncertain; disabling it causes alerts per note to jump from 9.88 to 12.08 (+2.20 alerts/note) as marginal extractions trigger review.
