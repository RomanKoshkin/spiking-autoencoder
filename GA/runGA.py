import os, time
import subprocess
import numpy as np
import copy, sys
import pandas as pd
import pickle

nChildren = 16
nGenerations = 200
p_elect = 0.3

# the keys to be ignored in the dna
immutable_genes = ["program", "script", "T", "K", "gen", "cell_id"]

dna = {
    "program": "python",
    "script": "sim.py",
    "Cp": 0.09636,
    "Cd": 0.00599,
    "tpp": 12.315,
    "tpd": 83.681,
    "taustf": 12.625,
    "taustd": 50.141,
    "alpha": 61.18,
    "gen": 0,
    "cell_id": 0
}

sigma = {
    "Cp": 0.04,
    "Cd": 0.01,
    "tpp": 4.0,
    "tpd": 4.0,
    "taustf": 15.0,
    "taustd": 15.0,
    "alpha": 12.0,
}

lim = {
    "Cp": [0.001, 0.25],
    "Cd": [0.0005, 0.25],
    "tpp": [5, 100],
    "tpd": [5, 100],
    "taustf": [10, 1350],
    "taustd": [10, 1350],
    "alpha": [1, 150],
}

mutable_genes = [i for i in dna.keys() if i not in immutable_genes]


def runGeneration(DNA, genID):
    PROC = []
    gen = 0
    for dna in DNA:
        PROC.append(subprocess.Popen([str(dna[i]) for i in dna.keys()]))
    C = np.zeros((nChildren,))
    # loop until every child has exited
    while C.sum() != nChildren:
        time.sleep(1)
        for i, proc in enumerate(PROC):
            if proc.poll() == 0:
                C[i] = 1
        sys.stdout.write(str(C.astype(int)))
        sys.stdout.flush()
        sys.stdout.write("\b" * len(str(C)))
    print()


def createMutant(dna_):
    global mutable_genes
    dna = copy.deepcopy(dna_)
    for i in mutable_genes:
        dna[i] += np.random.randn() * sigma[i]
        if dna[i] > lim[i][1]:
            dna[i] = lim[i][1]
        if dna[i] < lim[i][0]:
            dna[i] = lim[i][0]
    return dna


def getFittesGenes():
    global p_elect, nChildren
    df = pd.read_csv('data/DNA.csv')
    return df[df.gen == df.gen.max()].sort_values(by='loss').reset_index(drop=True)[:int(nChildren * p_elect)]


FITNESS, DNA, genID = [], [], 0

# make initial population
for chID in range(nChildren):
    dna['cell_id'] = chID
    DNA.append(createMutant(dna))

# delete the old DNA log, create and empty one with a header
if 'DNA.csv' in os.listdir('data'):
    os.remove('data/DNA.csv')

with open('data/DNA.csv', 'a') as f:
    header = "".join(f"{k}," for k, v in dna.items()) + 'loss\n'  # drop the last comma, add return
    f.write(header)

# run evolution
for genID in range(nGenerations):
    # remove old spike files
    for fname in [f for f in os.listdir('data') if f.startswith('spike_times_')]:
        os.remove(f'data/{fname}')
    runGeneration(DNA, genID)  # run one generation
    topDNAinGen = getFittesGenes()  # get the top ones in a DataFrame
    topDNAinGen = topDNAinGen.to_dict(orient='records')  # convert to list of dicts

    # breed from the fittest
    DNA1 = []
    for chID in range(nChildren):
        parentDNA = copy.deepcopy(np.random.choice(topDNAinGen))
        del parentDNA['loss']
        parentDNA['cell_id'] = chID  # set child ID
        parentDNA['gen'] = genID  # increment generation ID
        DNA1.append(createMutant(parentDNA))  # mutate and store
    DNA = DNA1
