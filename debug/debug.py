import os, sys

sys.path.append('/home/roman/CODE/spiking-autoencoder/')
print(f'PID: {os.getpid()}')
from modules.cClasses_10b import cClassOne
from modules.utils import *

import sys, time, threading, json

print(f'Python version: {sys.version_info[0]}')
import os, subprocess, pickle, shutil, itertools, string, warnings
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
import scipy
from scipy import signal
from numba import njit, jit, prange
from numba.typed import List

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from termcolor import cprint
from pprint import pprint

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

from tqdm import trange, tqdm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

HAGA = True
cell_id = 6347
pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
# pathToBucket = '/home/roman/bucket/slurm_big_HAGAwoFD/'
dt = 0.01
NEo = 0
NE = 400 + NEo
NI = 80  #157

params = load_config('../configs/config_1.yaml')

# override selected params to values that are good for BOTH withFD AND woFD self-organization:

case = 3

with open(f'../data/UU_case{case}.npy', 'rb') as f:
    UU = np.load(f)

for k, v in {
        'HAGA': True,
        'U': np.mean(UU),
        'Cp': 0.14,
        'Cd': 0.02,
        'taustf': 350.0,
        'taustd': 250.0,
        'tpp': 15.1549,
        'tpd': 120.4221
}.items():
    params[k] = v

# build model
m = cClassOne(NE, NI, NEo, cell_id, home='/home/roman/CODE/spiking-autoencoder/modules')
m.setParams(params)
m.saveSpikes(1)

# restore model weight, relprobs, spike states etc from checkpoint of the given experiment
experiment_id = 9
stimDur = 16000
nneur = 10
wide = True

m.sim(1000000)

cprint('success!', color='yellow')