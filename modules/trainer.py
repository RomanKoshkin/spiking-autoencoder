from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import trange
from modules.environment import InfinitePong
from modules.constants import bar_format
from modules.utils import *


class Trainer(object):

    def __init__(self, m, NE, stim_electrodes):
        self.action = 0
        self.NE = NE
        self.m = m
        self.stim_electrodes = stim_electrodes
        self.env = InfinitePong(visible=False)

        self.WgtEvo = []
        self.FEvo = []
        self.DEvo = []
        self.MOD = []

    def on_reward(self, reward):
        x = np.ones((self.NE,)).astype('int32')
        self.m.setStim(x)

        if reward == -1:
            stimpat = np.zeros((400,), dtype=np.float64)
            stim_electrodes = np.random.choice(self.stim_electrodes, 4, replace=False)
            stimpat[stim_electrodes] = 2
            stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1)

            for i in range(16):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(500)

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(2000)
            return stimpat

        if reward == 1:
            stimpat = np.zeros((400,), dtype=np.float64)
            stimpat[self.stim_electrodes] = 2
            stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1)

            for i in range(10):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(500)

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(500)
            return stimpat
        return None

    def on_sensory(self, im):
        x1 = np.zeros((self.NE,), dtype=np.float64)
        x1[:400] = im.flatten()
        self.m.setStimIntensity(x1)
        self.m.sim(5000)
        return im

    def train(self, nsteps):
        pbar = trange(nsteps, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)  # 2000 for 400 s
        for i in pbar:
            try:
                action = np.random.choice([-1, 0, 1])
                im, xpos, ypos, reward = self.env.step(action=self.action, gauss=True)
                if reward == 0:
                    _ = self.on_sensory(im)
                else:
                    _ = self.on_reward(reward)

                if i % 1 == 0:
                    W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
                    w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
                    map_dict = {j: i for i, j in enumerate(newids)}

                    t = self.m.getState().t
                    pbar.set_description(
                        f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, a: {action}, rw: {reward}'
                    )
                    self.WgtEvo.append(calcAssWgts(t, self.m, labels))
                    self.FEvo.append(calcF(t, self.m, labels))
                    self.DEvo.append(calcD(t, self.m, labels))
                    self.MOD.append(mod)
            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
