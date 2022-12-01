"""Configuration for get_features.py"""

from utils import AttrDict

# Parameters below are for the FireProt dataset
# Please change args if you plan to use a custom dataset
args = AttrDict({
    'raw': 'data/fireprot.csv',  # Path to raw data
    'output': 'data/features.csv',  # Path to the generated dataset
    'mean': 'data/mean.csv',  # Path to the mean values of the columns in features
    'std': 'data/std.csv',  # Path to the std values of the columns in features
    'chain': 'pdb_id',  # Column name for protein chains in raw data
    'mutation_location': 'position',  # Column name for mutation locations in raw data
    'wild_type': 'wild_type',  # Column name for wild type amino acids in raw data
    'mutant': 'mutation',  # Column name for mutant amino acids in raw data
    'ddg': 'ddG'  # Column name for ddG values in raw data
})
