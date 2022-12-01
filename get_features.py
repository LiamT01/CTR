"""Generates features"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from rawdata_config import args


def net_volume(wild, mutant):
    lookup = {'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9, 'G': 60.1, 'H': 153.2, 'I': 166.7,
              'K': 168.6, 'L': 166.7, 'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
              'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6}
    return lookup.get(mutant, 0) - lookup.get(wild, 0)


def net_hydrophobicity(wild, mutant):
    lookup = {'A': 0, 'C': 0.49, 'D': 2.95, 'E': 1.64, 'F': -2.2, 'G': 1.72, 'H': 4.76, 'I': -1.56,
              'K': 5.39, 'L': -1.81, 'M': -0.76, 'N': 3.47, 'P': -1.52, 'Q': 3.01, 'R': 3.71,
              'S': 1.83, 'T': 1.78, 'V': -0.78, 'W': -0.38, 'Y': -1.09}
    return lookup.get(mutant, 0) - lookup.get(wild, 0)


def net_flexibility(wild, mutant):
    lookup = {'A': 1, 'C': 3, 'D': 18, 'E': 54, 'F': 18, 'G': 1, 'H': 36, 'I': 9,
              'K': 81, 'L': 9, 'M': 27, 'N': 36, 'P': 2, 'Q': 108, 'R': 81,
              'S': 3, 'T': 3, 'V': 3, 'W': 36, 'Y': 18}
    return lookup.get(mutant, 0) - lookup.get(wild, 0)


def mutation_hydrophobicity(wild, mutant):
    def lookup(aa):
        if aa in ['A', 'C', 'F', 'I', 'L', 'M', 'V', 'W']:
            return 0
        if aa in ['G', 'H', 'P', 'S', 'T', 'Y']:
            return 1
        if aa in ('N', 'D', 'Q', 'E', 'K', 'R'):
            return 2

    return lookup(wild) * 3 + lookup(mutant)


def mutation_polarity(wild, mutant):
    def lookup(aa):
        if aa in ['H', 'K', 'R']:
            return 0
        if aa in ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W']:
            return 1
        if aa in ['N', 'Q', 'S', 'T', 'Y']:
            return 2
        if aa in ['D', 'E']:
            return 3

    return lookup(wild) * 4 + lookup(mutant)


def size(wild, mutant):
    def lookup(aa):
        if aa in ['C', 'D', 'N', 'P', 'T']:
            return 0
        if aa in ['E', 'H', 'Q', 'V']:
            return 1
        if aa in ['I', 'K', 'L', 'M', 'R']:
            return 2
        if aa in ['F', 'W', 'Y']:
            return 3
        if aa in ['A', 'G', 'S']:
            return 4

    return lookup(wild) * 5 + lookup(mutant)


def hydrogen_bond(wild, mutant):
    def lookup(aa):
        if aa in ['A', 'C', 'F', 'G', 'I', 'L', 'M', 'P', 'V']:
            return 0
        if aa in ['K', 'R', 'W']:
            return 1
        if aa in ['H', 'N', 'Q', 'S', 'T', 'Y']:
            return 2
        if aa in ['D', 'E']:
            return 3

    return lookup(wild) * 4 + lookup(mutant)


def chemical_property(wild, mutant):
    def lookup(aa):
        if aa in ['H', 'K', 'R']:
            return 0
        if aa in ['N', 'Q']:
            return 1
        if aa in ['D', 'E']:
            return 2
        if aa in ['C', 'M']:
            return 3
        if aa in ['S', 'T']:
            return 4
        if aa in ['F', 'W', 'Y']:
            return 5
        if aa in ['A', 'G', 'I', 'L', 'P', 'V']:
            return 6

    return lookup(wild) * 7 + lookup(mutant)


def mutation_type(wild, mutant):
    lookup = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
    return lookup.index(wild) * 21 + lookup.index(mutant)


def positional_features(chain, mutation_location, wild):
    def aa2int(aa):
        lookup = "ACDEFGHIKLMNPQRSTVWYX"
        return lookup.index(aa) + 1

    mutation_location -= 1
    features = []
    sequence = ""
    with open(f'data/seq/{chain}', 'r') as f:
        for line in f:
            string = line.strip()
            if string[0] != ">":
                sequence += string
    for i in sequence:
        if i.upper() not in "ACDEFGHIKLMNPQRSTVWYX":
            print(f"Error: {i} not recognized!")
            sys.exit()
        if i.upper() == 'X':
            print("Warning: X detected!")
    if wild.upper() != sequence[mutation_location]:
        print("Error: Wild type is not consistent with the sequence at the mutation location!")
        sys.exit()

    start_idx, end_idx = mutation_location - 5, mutation_location + 6

    # Pad 0's if there are not enough amino acids before the wild type
    if start_idx < 0:
        features.extend([0] * -start_idx)
        start_idx = 0

    features.extend([aa2int(aa) for aa in sequence[start_idx: min(end_idx, len(sequence))]])

    # Pad 0's if there are not enough amino acids after the wild type
    if end_idx > len(sequence):
        features.extend([0] * (end_idx - len(sequence)))
    return features


def check_seq(chain):
    return os.path.exists(f'data/seq/{chain}')


def check_pssm(chain):
    return os.path.exists(f'data/pssm/{chain}.pssm')


def pssm(chain, mutation_location):
    mutation_location -= 1
    windows = 9
    pssm = []
    lines = []
    with open(f'data/pssm/{chain}.pssm', 'r') as f:
        for line in f:
            lines.append(line.strip())

    start_idx = 0

    while len(lines[start_idx]) == 0 or lines[start_idx][0] != '1':
        start_idx += 1

    seq_length = int(lines[-7].split()[0])

    for j in range(mutation_location - windows // 2, mutation_location + windows // 2 + 1):
        # Pad 0's if out of bound
        if j < 0 or j > seq_length - 1:
            pssm.extend([0] * 20)
        else:
            pssm.extend([float(item) for item in lines[start_idx + j].split()[2: 22]])
    return pssm


def psepssm(chain):
    _lambda = 8
    pssm = []

    with open(f'data/pssm/{chain}.pssm', 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) > 0 and line[0].isnumeric() and line[1].isupper():
                line = np.array(line[2: 22], dtype=float)
                line_sigmoid = 1 / (1 + np.exp(-line))
                pssm.append(line_sigmoid)

    pssm = np.array(pssm)

    avg_pssm = np.sum(pssm, axis=0) / pssm.shape[0]
    features = avg_pssm.tolist()

    diff_pssm = []
    for i in range(1, _lambda + 1):
        diff_pssm.extend(np.sum(np.square(pssm[:-i] - pssm[i:]), axis=0) / (pssm.shape[0] - i))

    features.extend(diff_pssm)

    return features


def three2one(three):
    lookup = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
              'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
              'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    return lookup[three]


def get_feature_vector(chain, mutation_location, wild, mutant):
    if len(wild) == 3:
        wild = three2one(wild)
    if len(mutant) == 3:
        mutant = three2one(mutant)
    features = [net_volume(wild, mutant), net_hydrophobicity(wild, mutant), net_flexibility(wild, mutant),
                mutation_hydrophobicity(wild, mutant), mutation_polarity(wild, mutant), mutation_type(wild, mutant),
                size(wild, mutant), hydrogen_bond(wild, mutant), chemical_property(wild, mutant)]
    features += positional_features(chain, mutation_location, wild)
    features += pssm(chain, mutation_location)
    features += psepssm(chain)

    return features


def generate_file():
    data = pd.read_csv(args.raw)

    features = []
    targets = []

    for i in tqdm(range(len(data))):
        chain = data[args.chain][i]
        mutation_location = data[args.mutation_location][i]
        wild = data[args.wild_type][i]
        mutant = data[args.mutant][i]
        target = data[args.ddg][i]

        if check_seq(chain) and check_pssm(chain):
            feat = get_feature_vector(chain, mutation_location, wild, mutant)
            features.append(feat)
            targets.append(target)
        else:
            print(f'Error: Sequence file or PSSM file for chain {chain} (row index {i} in raw data) not found!')
            sys.exit()

    features = pd.DataFrame(features, dtype='float')

    features.mean(axis=0).to_csv(args.mean, index=False)
    features.std(axis=0).to_csv(args.std, index=False)

    features = features.apply(lambda x: (x - x.mean()) / x.std())
    features['targets'] = targets

    features.to_csv(args.output, index=False)
    print(f'Successfully generated a dataset of {len(features)} records.')


def generate_one_vector(chain, mutation_location, wild, mutant):
    feat = None

    if check_seq(chain) and check_pssm(chain):
        feat = get_feature_vector(chain, mutation_location, wild, mutant)
        feat = np.array(feat)
    else:
        print(f'Error: Sequence file or PSSM file for chain {chain} (row index {i} in raw data) not found!')
        sys.exit()

    mean = pd.read_csv(args.mean).to_numpy().flatten()
    std = pd.read_csv(args.std).to_numpy().flatten()
    feat = (feat - mean) / std
    return feat


if __name__ == '__main__':
    generate_file()
