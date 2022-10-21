from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import trange
from modules.environment import InfinitePong
from modules.constants import bar_format
from modules.utils import *


class Trainer(object):

    def __init__(self, m, NE, stim_electrodes):
        self.NE = NE
        self.m = m
        self.stim_electrodes = stim_electrodes
        self.env = InfinitePong(visible=False)

        self.WgtEvo = []
        self.FEvo = []
        self.DEvo = []
        self.MOD = []
        self.prev_degree = None

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

    def _snapshot(self, i, im, activity, action, reward, pbar):
        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
        w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
        t = self.m.getState().t
        pbar.set_description(
            f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, a: {action}, rw: {reward}')

        self.WgtEvo.append(calcAssWgts(t, self.m, labels))
        self.FEvo.append(calcF(t, self.m, labels))
        self.DEvo.append(calcD(t, self.m, labels))
        self.MOD.append(mod)

        indegrees = W[:400, :400].sum(axis=1).reshape(20, 20)
        outdegrees = W[:400, :400].sum(axis=0).reshape(20, 20)
        degree = indegrees + outdegrees
        if self.prev_degree is None:
            degree_diff = np.ones((20, 20)) * 1e-9
        else:
            degree_diff = degree - self.prev_degree
        self.prev_degree = np.copy(degree)

        _, ax = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
        _ = ax[0].imshow(im)
        _ = ax[1].imshow(activity)
        _ = ax[2].imshow(degree / degree.max())
        _ = ax[3].imshow(degree_diff / degree_diff.max())
        _title = f'{self.m.getState().t:.2f} step: {self.env.env.stepid} mod: {mod:.2f}, action {action}'
        _ = ax[1].set_title(_title)
        _ = plt.savefig(f'../assets/activity_{i:05d}.png')
        _ = plt.close()

    def train(self, nsteps):

        removeFilesInFolder('../assets/')  # delete old frames

        pbar = trange(nsteps, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)  # 2000 for 400 s
        for i in pbar:
            try:
                action = np.random.choice([-1, 0, 1])
                im, xpos, ypos, reward = self.env.step(action=action, gauss=True)
                if reward == 0:
                    _ = self.on_sensory(im)
                else:
                    _ = self.on_reward(reward)

                activity = self.m.getRecents()[:400].reshape(20, 20)

                self._snapshot(i, im, activity, action, reward, pbar)

            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
