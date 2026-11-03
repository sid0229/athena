This is the DrugBank dataset for the paper "HDN-DDI: a novel framework for predicting drug-drug interactions using hierarchical molecular graphs and enhanced dual-view representation learning" (Paper | [Code](https://github.com/jcsun-00/HDN-DDI.git))

The DrugBank dataset was proposed by [Wishart et al](https://doi.org/10.1093/nar/gkx1037). and has been iterated in multiple versions. The full dataset can be found [here](https://go.drugbank.com/).

The version used in the HDN-DDI paper is the same as that used in the [BDN-DDI](https://doi.org/10.1016/j.compbiomed.2023.107340), [DSN-DDI](https://doi.org/10.1093/bib/bbac597) and other papers. Specifically, the DrugBank used by DDI prediction methods is comprises 1,706 drugs with 86 types of interactions, yielding 191,808 positive DDI instances.

If you want to use this dataset to verify the performance of HDN-DDI, please download the dataset and place the files according to the following structure:
```
- drugbank_test /
    - DrugBank /
        - cold_start / ...
        - warm_start / ...
        - ddis.csv
        - drug_smiles.csv
        - id_data_dict_dsn_full_connect.pkl
    - transductive_test.py
    - inductive_test.py
    - ...
```

