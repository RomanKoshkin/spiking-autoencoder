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
        removeFilesInFolder('../assets/')  # delete old frames

    def on_reward(self):
        x = np.ones((self.NE,)).astype('int32')
        self.m.setStim(x)

        if self.reward == -1:
            # FIXME: this must be an UNpredictable stim
            stimpat = np.zeros((400,), dtype=np.float64)
            if len(self.stim_electrodes) > 3:
                stim_electrodes = np.random.choice(self.stim_electrodes, 4, replace=False)
                stimpat[stim_electrodes] = 2
                stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1)
            else:
                pass

            for i in range(16):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(500)
                self._snapshot(x1[:400].reshape(20, 20))

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(2000)
                self._snapshot(x1[:400].reshape(20, 20))
            return stimpat

        if self.reward == 1:
            # FIXME: this must be a predictable stim
            stimpat = np.zeros((400,), dtype=np.float64)
            if len(self.stim_electrodes) > 3:
                stim_electrodes = np.random.choice(self.stim_electrodes, 4, replace=False)
                stimpat[stim_electrodes] = 2
                stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1)
            else:
                pass

            for i in range(10):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(500)
                self._snapshot(x1[:400].reshape(20, 20))

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(500)
                self._snapshot(x1[:400].reshape(20, 20))
            return stimpat
        return None

    def on_sensory(self, im):
        x1 = np.zeros((self.NE,), dtype=np.float64)
        x1[:400] = im.flatten()
        self.m.setStimIntensity(x1)
        self.m.sim(5000)
        self._snapshot(im)
        return im

    def _snapshot(self, im):
        activity = self.m.getRecents()[:400].reshape(20, 20)
        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
        w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
        t = self.m.getState().t
        self.pbar.set_description(
            f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, a: {self.action}, rw: {self.reward}')

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

        X, Y, U, V = compute_weight_bias(W[:400, :400])

        _, ax = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
        _ = ax[0].imshow(im)
        _ = ax[0].set_title('stimulus')

        _ = ax[1].imshow(activity)
        ax[1].quiver(X, Y, U, V, angles='xy', scale=30, color='red', width=0.005)
        _ = ax[1].set_title('low-passed activity')

        _ = ax[2].imshow(degree / degree.max())
        _ = ax[2].set_title('degree')

        _ = ax[3].imshow(degree_diff / degree_diff.max())
        _ = ax[3].set_title('deriv. of degree')

        _title = (f'{self.m.getState().t:.2f} step: {self.env.env.stepid}' +
                  f'mod: {mod:.2f}, action {self.action}, rw: {self.reward}')
        _ = ax[1].set_title(_title)
        _ = plt.savefig(f'../assets/activity_{int(self.m.getState().t*100):09d}.png')
        _ = plt.close()

    def train(self, nsteps):

        self.pbar = trange(nsteps, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)  # 2000 for 400 s
        for self.i in self.pbar:
            try:
                self.action = np.random.choice([-1, 0, 1])
                im, xpos, ypos, self.reward = self.env.step(action=self.action, gauss=True)
                if self.reward == 0:
                    _ = self.on_sensory(im)
                else:
                    _ = self.on_reward()

            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
