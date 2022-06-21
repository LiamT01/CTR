# CTR

**C**lustered **T**ree **R**egression to Learn Protein Energy Change with Mutated Amino Acid

Here we provide code to implement CTR for predicting change in protein folding free energy upon single-point amino acid mutation.

The FireProt dataset can be found at ./data/fireprot.csv.

Please cite:

```Latex
@article{tu2022clustered,
  title={Clustered tree regression to learn protein energy change with mutated amino acid},
  author={Tu, Hongwei and Han, Yanqiang and Wang, Zhilong and Li, Jinjin},
  journal={Briefings in Bioinformatics},
  year={2022}
}
```

## File Organization

- ./data: Contains data used in CTR.
  - ./data/out: Contains output files by BLAST. No output files would be used by CTR, so this folder was left empty initially.
  - ./data/pssm: Contains PSSM files by BLAST. We have already generated PSSM files for the FireProt dataset. You need to run BLAST and generate PSSM files if you plan to use a custom dataset.
  - ./data/seq: Contains sequence files for proteins in a dataset. We have collected all sequence files for the FireProt dataset. You need to collect protein sequences and name the files with the protein names if you plan to use a custom dataset.
  - ./fireprot.csv: The FireProt dataset.
- ./get_pssm.sh: A shell script to run BLAST and generate PSSM files. You need to modify paths in the script beforehand.
- ./get_features.py: A Python script to generate a feature file (.csv). You need to check/modify the configuration in ./rawdata_config.py beforehand.
- ./train_test.py: A python script to train and test a model using a dataset (FireProt or custom). You need to check/modify the configuration in ./model_config.py beforehand.
- ./rawdata_config.py: Configuration for get_features.py.
- ./model_config.py: Configuration for train_test.py.
- ./utils.py: Helper functions/classes.
- ./requirements.txt: Package specification for environment setup.

## Usage

1. Clone this repository.
2. Set up an environment as per ./requirements.txt.
3. (Optional) Collect sequence files for a custom dataset.
4. (Optional) Run ./get_pssm.sh to generate PSSM files for a custom dataset. You need to install BLAST (https://blast.ncbi.nlm.nih.gov/Blast.cgi) and download the non-redundant (nr) database beforehand.
5. (Optional) Modify ./rawdata_config.py for your custom dataset.
6. Run ./get_features.py to generate a feature file (.csv).
7. (Optional) Modify ./model_config.py.
8. Run ./train_test.py to train and test a model.

## Notes

The prediction accuracy may differ slightly from that in the paper due to the version of XGBoost in use.

## License

Distributed under the Apache 2.0 License. See LICENSE for more information.

## Acknowledgements

The work is supported by the National Key R&D Program of China (No. 2021YFC2100100), the National Natural Science Foundation of China (No. 21901157), the Shanghai Science and Technology Project (No. 21JC1403400), and the SJTU Global Strategic Partnership Fund (2020 SJTU-HUJI).
