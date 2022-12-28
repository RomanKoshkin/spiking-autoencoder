import argparse, warnings
from modules.cClasses_10b import cClassOne
from modules.utils import *
import numpy as np
import pandas as pd
from termcolor import cprint
from tqdm import trange

warnings.filterwarnings("ignore")

cell_id = 6347
pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
params = load_config('/home/roman/CODE/spiking-autoencoder/configs/config_sequential.yaml')

dt = params['dt']
NEo = params['NEo']
NE = params['NE']
NM = params['NM']
NI = params['NI']
case = params['case']
wide = params['wide']

# overrides
parser = argparse.ArgumentParser(description="Dishbrain simulator")
parser.add_argument("--HAGA", type=int, default=None)
parser.add_argument("--astrocytes", type=int, default=None)
parser.add_argument("--runid", type=int, default=None)
parser.add_argument("--stim_strength", type=float, default=None)
parser.add_argument("--nass", type=int, default=None)
parser.add_argument("--rotate_every_ms", type=int, default=None)
parser.add_argument("--stim_time_ms", type=int, default=None)
parser.add_argument("--total_time_ms", type=int, default=None)
args = parser.parse_args()

# NOTE: case is always 3, wide is always True
UU = simple_make_UU(NE, case, wide)

# update the params dict with overrides
for k, v in {
        'HAGA': bool(args.HAGA),
        'symmetric': True,
        'U': np.mean(UU),
        'Cp': 0.14,
        'Cd': 0.02,
        'taustf': 350.0,
        'taustd': 250.0,
        'tpp': 15.1549,
        'tpd': 120.4221,
}.items():
    params[k] = v

# build model and set params
m = cClassOne(NE, NI, NEo, cell_id)
m.setParams(params)
m.saveSpikes(0)

# set random UU
if bool(args.astrocytes):
    m.setUU(np.ascontiguousarray(UU))

m.set_mex(0.3)
m.set_hEhI(1.0, 1.0)
m.set_STDP(True)
m.set_homeostatic(True)

stimulator = Stimulator(
    m,
    stim_strength=args.stim_strength,
    nass=args.nass,
    rotate_every_ms=args.rotate_every_ms,
)

# make a header for the thetas file
d = dict()
d['t'] = 0
for k, pat in enumerate(stimulator.patterns):
    d[k] = 0
header = ",".join([str(k) for k in d.keys()]) + "\n"
with open('data/thetas', 'w') as f:
    f.write(header)

pbar = trange(2000)
t = m.getState().t

FD = []
for i in pbar:
    if t > args.total_time_ms:
        break
    try:
        for j in range(100):
            # NOTE: we record spikes +- 5s around the stimulus offset
            if m.getState().t < args.stim_time_ms:
                stimulator.check_if_rotate_stim()
                m.sim(200)
            else:
                stimulator.sham_stim()
                m.sim(200)

            # if the time is right, save spikes, and other stats
            if (t > args.stim_time_ms - 5000) and (t < args.stim_time_ms + 5000):
                m.saveSpikes(1)

                # save mean assembly-wise thetas
                theta = m.getTheta()
                t = m.getState().t
                d = dict()
                d['t'] = np.round(t, 2)
                for k, pat in enumerate(stimulator.patterns):
                    d[k] = theta[pat].mean()
                with open('data/thetas', 'a') as f:
                    f.write(",".join([str(v) for v in d.values()]) + "\n")

                fd = m.getF() * m.getD()
                inh = np.copy(m.getUinh())
                exc = np.copy(m.getUexc())
                d = {'spiketime': m.getState().t}
                for k, pat in enumerate(stimulator.patterns):
                    d[f'inh_{k}'] = np.mean(inh[pat])
                    d[f'exc_{k}'] = np.mean(exc[pat])
                    d[f'fd_{k}'] = np.mean(fd[pat])
                FD.append(copy.deepcopy(d))
            else:
                m.saveSpikes(0)

        if i % 1 == 0:
            W = np.copy(m.getWeights())[:NE, :NE]
            w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
            map_dict = {j: i for i, j in enumerate(newids)}

            t = m.getState().t
            nass = len(np.unique(labels))
            with open('data/simres', 'a') as f:
                f.write(
                    f'{t:.2f},{mod:.3f},{args.nass},{nass},{bool(args.HAGA)},{bool(args.astrocytes)},{args.runid}\n')

    except KeyboardInterrupt:
        cprint('User interrupt', color='green')
        break

    pd.DataFrame(FD).to_csv('data/inhib_exc.csv')
