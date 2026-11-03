# Data sources and licences

Included in this repository (redistributed with attribution, non-commercial academic use):

| Folder | Source | Licence |
|---|---|---|
| `raw/drugbank_benchmark/` | DeepDDI DrugBank benchmark as packaged by HDN-DDI (github.com/jcsun-00/HDN-DDI); Wishart et al., *Nucleic Acids Res.* 2018 | DrugBank data: CC BY-NC 4.0 |
| `raw/drugbank_kaggle/` | DrugBank drug–drug interaction extract (Kaggle) | DrugBank data: CC BY-NC 4.0 |
| `raw/ddinter/` | DDInter, Xiong et al., *Nucleic Acids Res.* 2022 (ddinter.scbdd.com) | CC BY-NC-SA 4.0 |
| `raw/mimic_iii_demo/` | MIMIC-III Clinical Database Demo v1.4, PhysioNet (Johnson et al., 2019) | Open Data Commons ODbL 1.0 |
| `raw/rxnorm/RxNorm_full_prescribe_*.zip` | RxNorm Current Prescribable Content, US National Library of Medicine | No licence required; NLM attribution |
| `raw/wikidata/` | Wikidata drug labels, aliases and identifiers | CC0 |
| `processed/` | Built by `scripts/build_lexicon.py`, `scripts/build_kb.py`, `scripts/train_ddi_xgboost.py` from the above | As the sources above |

DrugBank- and DDInter-derived files are for **non-commercial** use only. A deployed version should
replace DrugBank with an openly licensed source such as DrugCentral (see `docs/upgrade_experiments.md`).

Not included:

| Data | Why | How to get it |
|---|---|---|
| n2c2 2018 Track 2 (`raw/n2c2_2018_track2/`) | n2c2 Data Use Agreement forbids redistribution | Request access at portal.dbmi.hms.harvard.edu (n2c2 NLP Research Data Sets), then place the three zip files in `data/raw/n2c2_2018_track2/` |
| LLM output cache (`processed/llm_cache/`) | Derived from n2c2 note text | Regenerated automatically by `scripts/warm_llm_cache.py` |
| RxNorm RRF files (`raw/rxnorm/rrf/`) | `RXNREL.RRF` exceeds GitHub's 100 MB limit | `unzip -j data/raw/rxnorm/RxNorm_full_prescribe_*.zip 'rrf/RXNCONSO.RRF' 'rrf/RXNREL.RRF' -d data/raw/rxnorm/rrf/` |
| FDA FAERS 2026 Q2 (`raw/faers/`) | `DRUG26Q2.txt` exceeds 100 MB; optional signal only | fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html (ASCII, 2026 Q2) |
