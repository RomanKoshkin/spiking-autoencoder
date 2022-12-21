import sys, json, time

sys.path.append('../')
import numpy as np
from modules.cClasses_10b import cClassOne
from modules.constants import bar_format
from modules.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
params = load_config('../configs/config_sequential.yaml')
immutable_genes = ["program", "script", "gen", "cell_id"]

dna = {
    "program": "python",
    "script": "sim.py",
    "Cp": None,
    "Cd": None,
    "tpp": None,
    "tpd": None,
    "taustf": None,
    "taustd": None,
    "alpha": None,
    "gen": None,
    "cell_id": None,
}

# get dna
for i, (k, v) in enumerate(dna.items()):
    if not k in ['program', 'script']:
        if k in ['gen', 'cell_id']:
            dna[k] = int(sys.argv[i - 1])  # NOTE: arg0 is fname, arg1 is T etc
        else:
            dna[k] = float(sys.argv[i - 1])  # NOTE: arg0 is fname, arg1 is T etc

# override the configs with mutated DNA
for k, v in dna.items():
    if k not in immutable_genes:
        params[k] = v

dt = params['dt']
NEo = params['NEo']
NE = params['NE']
NM = params['NM']
NI = params['NI']
case = params['case']
wide = params['wide']
cell_id = dna['cell_id']

# build model and set params
m = cClassOne(NE, NI, NEo, cell_id, home="/home/roman/CODE/spiking-autoencoder/modules/")
m.setParams(params)
m.saveSpikes(1)

# hard-wire 4 cell assemblies
step = 100
w = m.getWeights()
for I in range(0, NE, step):
    for i in range(I, I + step):
        for j in range(I, I + step):
            if w[i, j] > 0.001:
                w[i, j] = 1.9

for I in range(0, NE - step, step):
    for i in range(I, I + step):
        for j in range(I, I + step):
            if w[i + step, j] > 0.001:
                w[i + step, j] = 0.6

for i in range(0, step):
    for j in range(NE - step, 400):
        if w[i, j] > 0.001:
            w[i, j] = 0.6

m.setWeights(w)

# set random UU
UU = simple_make_UU(NE, case, wide)
m.setUU(np.ascontiguousarray(UU))

WgtEvo, FEvo, DEvo, MOD = [], [], [], []
m.set_mex(0.3)
m.set_hEhI(1.0, 1.0)
m.set_STDP(True)
m.set_homeostatic(True)
# m.set_STDP(False)
# m.set_homeostatic(False)

stimulate = False
# stimulate = True

if stimulate:
    x = np.zeros((NE,)).astype('int32')
    x[NE - NEo:] = 1
    m.setStim(x)
    x1 = np.zeros((NE,)).astype(float)
    m.setStimIntensity(x1)
else:
    x = np.zeros((NE,)).astype('int32')
    m.setStim(x)
    x1 = np.zeros((NE,)).astype(float)
    m.setStimIntensity(x1)

for i in range(100):
    try:
        if stimulate:
            x = np.zeros((NE,)).astype('int32')
            x1 = np.zeros((NE,)).astype(float)
            x[NE - NEo:] = 1  # the trailing excitatory neurons are input neurons
            x1[NE - NEo:] = 1.0
            m.setStim(x)
            m.setStimIntensity(x1)
            m.sim(10000)

            # no stim interval
            x1 = np.zeros((NE,)).astype(float)
            m.setStimIntensity(x1)
            m.sim(10000)
        else:
            # no stim interval
            x1 = np.zeros((NE,)).astype(float)
            m.setStimIntensity(x1)
            m.sim(20000)

        if i % 1 == 0:
            W = np.copy(m.getWeights())[:NE, :NE]
            w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
            map_dict = {j: i for i, j in enumerate(newids)}

            t = m.getState().t
            WgtEvo.append(calcAssWgts(t, m, labels))
            FEvo.append(calcF(t, m, labels))
            DEvo.append(calcD(t, m, labels))
            MOD.append(mod)
    except KeyboardInterrupt:
        cprint('User interrupt', color='green')
        break

# read spikes
sp = pd.read_csv(f'data/spike_times_{cell_id}', delimiter=' ', header=None)
sp.columns = ['spiketime', 'neuronid']
st, en = sp.spiketime.max() - 5000, sp.spiketime.max()
sp = sp[(sp.spiketime > st) & (sp.spiketime < en)]

sp = processSpikes(m, sp, NE, NEo, NI)

correct_targets = [3, 0, 1, 2]

Z = getCondDistOfCAs(sp)
criterion = nn.CrossEntropyLoss(reduction='mean')
targets = torch.tensor(correct_targets)
try:
    loss = criterion(F.softmax(torch.from_numpy(Z)), targets).item()
except:
    loss = 100 + np.random.rand()

# write result (*dna, loss)
time.sleep(np.random.rand() * 4)
with open('data/DNA.csv', 'a') as f:
    line = "".join(f"{v}," for k, v in dna.items())
    line += f"{loss}\n"
    f.write(line)

plt.imshow(Z)
plt.savefig(f"data/{dna['gen']}_{dna['cell_id']}.png")
plt.close('all')