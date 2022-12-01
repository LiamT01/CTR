"""Evaluates a model"""

import argparse
import pickle

import numpy as np
import pandas as pd

from get_features import generate_one_vector

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', '-i', type=str, required=True, help='Path to input file')
parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to output file')
args = parser.parse_args()

if __name__ == '__main__':
    chains = []
    locations = []
    wilds = []
    mutants = []

    with open(args.input_path, 'r') as f:
        lines = f.read().split('\n')

    lines = [line for line in lines if len(line) > 0]
    cluster_path = lines[0]
    reg1_path = lines[1]
    reg2_path = lines[2]

    for line in lines[3:]:
        line = line.split()
        chains.append(line[0])
        locations.append(int(line[1]))
        wilds.append(line[2])
        mutants.append(line[3])

    data = []
    for chain, loc, wild, mutant in zip(chains, locations, wilds, mutants):
        data.append(generate_one_vector(chain, loc, wild, mutant))
    data = pd.DataFrame(data)

    cluster = pickle.load(open(cluster_path, "rb"))
    labels = cluster.predict(data)
    reg1 = pickle.load(open(reg1_path, "rb"))
    reg2 = pickle.load(open(reg2_path, "rb"))

    original = np.array(range(data.shape[0]))
    perm = np.concatenate([original[labels == 1],
                           original[labels == 0]])
    inverse = np.empty(data.shape[0]).astype(int)
    inverse[perm] = original

    preds = np.concatenate([reg1.predict(data.iloc[labels == 1]),
                            reg2.predict(data.iloc[labels == 0])])

    preds = preds[inverse].tolist()

    print('-' * 40)

    with open(args.output_path, 'w') as f:
        for pred in preds:
            f.write(f'{pred}\n')
            print(pred)
