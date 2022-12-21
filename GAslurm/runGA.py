import os, time
import subprocess
import numpy as np
import copy, sys
import pandas as pd
import pickle
from termcolor import cprint

nChildren = 100
nGenerations = 500
p_elect = 0.15
path_to_job_file_on_deigo = '/flash/FukaiU/roman/GA'
path_to_job_file_from_precision = '/home/roman/flash/GA'
output_folder_on_slurm = 'output'
job_file_name = 'start_job.sh'

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


def checkNumJobs():
    subproc = subprocess.Popen("ssh deigo 'squeue'", shell=True, stdout=subprocess.PIPE)
    ret = subproc.stdout.read().decode('UTF-8').split('\n')[1:-1]
    n_jobs_now = len(ret)
    msg = f'Number of jobs running: {n_jobs_now}'
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stdout.write("\b" * len(msg))
    return n_jobs_now


def MakeAJobFile(_dna, genID, cell_id):
    global path_to_job_file_from_precision, job_file_name, output_folder_on_slurm
    dna = copy.deepcopy(_dna)
    dna['gen'] = genID
    dna['cell_id'] = cell_id
    dna_str = " ".join(str(v) for k, v in dna.items()) + "\n"
    with open(f'{path_to_job_file_from_precision}/{job_file_name}', 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --job-name=JOBNAME1\n")
        f.writelines(f"#SBATCH --mail-user=roman.koshkin@oist.jp\n")
        f.writelines(f"#SBATCH --partition=short\n")
        f.writelines(f"#SBATCH --ntasks=1\n")
        f.writelines(f"#SBATCH --cpus-per-task=1\n")
        f.writelines(f"#SBATCH --mem-per-cpu=1g\n")
        f.writelines(f"#SBATCH --output=./{output_folder_on_slurm}/%j.out\n")
        f.writelines(f"#SBATCH --array=1-1\n")  # submit 4 jobs as an array, give them individual id from 1 to 4
        f.writelines(f"#SBATCH --time=0:15:00\n")
        f.writelines(dna_str)
        # f.writelines("python sim.py $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $1\n")


def runJob():
    global path_to_job_file_on_deigo, job_file_name
    commands = f'"cd {path_to_job_file_on_deigo} && sbatch {job_file_name}"'
    os.system(f"ssh deigo {commands}")


def runGeneration(DNA, genID):

    for cell_id, dna in enumerate(DNA):
        MakeAJobFile(dna, genID, cell_id)
        runJob()

    num_jobs = checkNumJobs()
    while num_jobs > 0:
        time.sleep(5)
        num_jobs = checkNumJobs()
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
    df = pd.read_csv(f'{path_to_job_file_from_precision}/data/DNA.csv')
    return df[df.gen == df.gen.max()].sort_values(by='loss').reset_index(drop=True)[:int(nChildren * p_elect)]


FITNESS, DNA, genID = [], [], 0

os.makedirs(output_folder_on_slurm, exist_ok=True)

# make initial population
for chID in range(nChildren):
    dna['cell_id'] = chID
    DNA.append(createMutant(dna))

# delete the old DNA log, create and empty one with a header
if 'DNA.csv' in os.listdir(f'{path_to_job_file_from_precision}/data'):
    os.remove(f'{path_to_job_file_from_precision}/data/DNA.csv')

with open(f'{path_to_job_file_from_precision}/data/DNA.csv', 'a') as f:
    header = "".join(f"{k}," for k, v in dna.items()) + 'loss\n'  # drop the last comma, add return
    f.write(header)

# run evolution
for genID in range(nGenerations):
    # remove old spike files
    for fname in [f for f in os.listdir(f'{path_to_job_file_from_precision}/data') if f.startswith('spike_times_')]:
        os.remove(f'{path_to_job_file_from_precision}/data/{fname}')
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
