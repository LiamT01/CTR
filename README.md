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
  volume={23},
  number={6},
  pages={bbac374},
  year={2022},
  publisher={Oxford University Press}
}
```

## File Organization

- `./data`: Contains data used in CTR.
  - `./data/out`: Contains output files by BLAST. No output files would be used by CTR, so this folder was left empty initially.
  - `./data/pssm`: Contains PSSM files by BLAST. We already generated PSSM files for the FireProt dataset. You may need to run BLAST and generate PSSM files if you plan to use a custom dataset.
  - `./data/seq`: Contains sequence files for proteins in a dataset. We collected all sequence files for the FireProt dataset. You may need to collect protein sequences and name the files as the protein names if you plan to use a custom dataset.
  - `./data/fireprot.csv`: The FireProt dataset.
  - `./data/eval_demo.txt`: A demo input file to evaluate a model using `./eval.py`.
- `./weights`: Contains saved model weights.
  - `./weights/cluster`: Contains the weights of the clustering models (K-means).
  - `./weights/reg1`: Contains the weights of the first regressors.
  - `./weights/reg2`: Contains the weights of the second regressors.
- `./eval.py`: A Python script to evaluate a model. A demo input file is at `./data/eval_demo.txt`.
- `./get_pssm.sh`: A shell script to run BLAST and generate PSSM files. You may need to modify the paths in the script beforehand.
- `./get_features.py`: A Python script to generate a feature file (.csv). You may need to modify the configuration in `./rawdata_config.py` beforehand.
- `./train_test.py`: A python script to train and test a model using a dataset (FireProt or custom). You may need to modify the configuration in `./model_config.py` beforehand.
- `./rawdata_config.py`: Configuration for `./get_features.py`.
- `./model_config.py`: Configuration for `./train_test.py`.
- `./utils.py`: Helper functions/classes.
- `./requirements.txt`: Package specification for environment setup.

## Preliminary

1. Clone this repository.
2. Set up an environment as per `./requirements.txt`.
3. (Optional) Collect sequence files for a custom dataset.
4. (Optional) Run `./get_pssm.sh` to generate PSSM files for a custom dataset. You need to install BLAST (<https://blast.ncbi.nlm.nih.gov/Blast.cgi>) and download the non-redundant (nr) database beforehand.
5. (Optional) Modify `./rawdata_config.py` for your custom dataset.
6. Run `./get_features.py` to generate a feature file (.csv).

## Training

1. (Optional) Modify `./model_config.py`.
2. Run ./train_test.py to train and test a model. Model weights of the clustering model, the first regressor, and the second regressor are saved at `./weights` after each fold in the cross-validation.

## Evaluation

1. Create an input file. A demo is at `./data/eval_demo.txt`.
   - The first three lines are paths to saved model weights (must be in the order of the clustering model, the first regressor, and the second regressor).

      ```Text
      weights/cluster/cluster_001
      weights/reg1/reg1_001
      weights/reg2/reg2_001
      ```

   - The following lines are input data, one protein sequence with one mutated amino acid per line. In each line, from left to right are: protein chain, mutation location (starting from 1 instead of 0), wild type, and mutant. If there are several mutation sites you want to predict for a single protein sequence, please enter one mutation site per line.

      ```Text
      1ACB 1 T W
      1ACB 3 F P
      1AG2 5 G E
      1AM7 2 V K
      ```

   - Blank lines are ignored.
2. Type the following in a terminal, where `-i` specifies the input, and `-o` specifies the output you want.

    ```Shell
    python eval.py -i path/to/input -o path/to/your/desired/output
    ```

    The predicted ddG values will be print to the screen and saved to the path specified by `-o`.
3. Notes
   - The protein sequences to be predicted must have their sequences saved at `./data/seq` and their PSSMs saved at `./data/pssm`. You may need to run BLAST (<https://blast.ncbi.nlm.nih.gov/Blast.cgi>) to get the PSSMs.

## Notes

The prediction accuracy may differ slightly from that in the paper due to the version of XGBoost in use.

## Trained Models

Trained model weights can be downloaded from:

1. Google Drive: <https://drive.google.com/file/d/1GaInKSR9vNEMG-B4jfPu3_6I4iEbJhlb/view?usp=sharing>
2. 百度网盘：<https://pan.baidu.com/s/1DM4qnYo85XbgjloKv7rGkQ>，提取码: 9ib0

## License

Distributed under the Apache 2.0 License. See LICENSE for more information.

## Acknowledgements

The work is supported by the National Key R&D Program of China (No. 2021YFC2100100), the National Natural Science Foundation of China (No. 21901157), the Shanghai Science and Technology Project (No. 21JC1403400), and the SJTU Global Strategic Partnership Fund (2020 SJTU-HUJI).
