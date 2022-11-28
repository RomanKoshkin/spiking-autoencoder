import sys, os

sys.path.append('../')

from modules.cClasses_10b import cClassOne
from modules.utils import *
from modules.constants import bar_format
from modules.environment import InfinitePong
from modules.trainer import Trainer
from modules.exp1 import Exp1

if __name__ == '__main__':

    cell_id = 6347
    pathToBucket = '/home/roman/bucket/slurm_big_HAGAwithFD/'
    params = load_config('config_1.yaml')

    dt = params['dt']
    NEo = params['NEo']
    NE = params['NE']
    NM = params['NM']
    NI = params['NI']
    case = params['case']
    wide = params['wide']

    # build model and set params

    m = cClassOne(NE, NI, NEo, cell_id)
    m.setParams(params)
    m.saveSpikes(1)

    m.set_mex(0.13)
    print(f'mex: {m.getState().mex}')
    m.set_hEhI(1.0, 1.0)
    m.set_STDP(True)
    m.set_homeostatic(True)

    # stim_electrodes = np.array([23, 50, 157, 204, 250, 304, 350, 357])
    stim_electrodes = np.array([])

    trainer = Trainer(m, NE, NM, stim_electrodes, config=params)

    trainer.env.env.paddle_on = True

    trainer.train(2000)