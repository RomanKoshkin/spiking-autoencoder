import imp
import numpy as np
from numba import njit
from LIFconstants import *

float_type = np.float64  # float32 is slower by 2x


@njit
def _clip_wts(w):
    global Jmin, Jmax
    if w < Jmin:
        w = Jmin
    if w > Jmax:
        w = Jmax
    return w


@njit
def make_weights():
    global N, NE, NI, sigJ, JEEinit, Jmin, Jmax, cEE, cII, cEI, cIE, JEE, JII, JEI, JIE

    w = np.zeros((N, N), dtype=float_type)

    # EE
    for i in range(NE):
        for j in range(NE):
            if np.random.rand() < cEE:
                w[i, j] = JEEinit * (1.0 + np.random.randn() * sigJ)
                w[i, j] = _clip_wts(w[i, j])

    # II
    for i in range(NI, N):
        for j in range(NI, N):
            if np.random.rand() < cII:
                w[i, j] -= JII

    # IE
    for i in range(NE, N):
        for j in range(NE):
            if np.random.rand() < cIE:
                w[i, j] -= JIE
    # EI
    for i in range(NI):
        for j in range(NI, N):
            if np.random.rand() < cEI:
                w[i, j] -= JEI

    return w


@njit
def init():
    w = make_weights()
    t = 0.0
    AP = np.zeros((N,), dtype=np.int64)

    # define neuron types in the network:
    neur_type_mask = np.zeros_like(AP)
    neur_type_mask[:NE] = 0
    neur_type_mask[NE:] = 1

    # initialize spikes
    exc_id = np.where(neur_type_mask == 1)[0]
    ons = np.random.choice(exc_id, int(0.4 * len(exc_id)), replace=False)
    for a in ons:
        AP[a] = 1

    # taus
    tau = np.zeros((N,), dtype=float_type)
    idx = np.where(neur_type_mask == 0)[0]
    for j in idx:
        tau[j] = TAU_EXCITATORY
    idx = np.where(neur_type_mask == 1)[0]
    for j in idx:
        tau[j] = TAU_INHIBITORY

    V = np.ones((N,), dtype=float_type) * EL

    # prohibit self-connections
    for i in range(N):
        w[i, i] = 0

    V = V * 0 + EL

    EPSILON = 0.001

    ampa = np.zeros((N, N), dtype=float_type)
    nmda = np.zeros((N, N), dtype=float_type)
    gaba = np.zeros((N, N), dtype=float_type)
    in_refractory = np.zeros((N,), dtype=float_type)  # !
    F = np.ones((N,), dtype=float_type) * U
    D = np.ones((N,), dtype=float_type)
    dV = np.zeros((N,), dtype=float_type)
    I_E = np.zeros((N,), dtype=float_type)
    I_I = np.zeros((N,), dtype=float_type)
    delayed_spike = np.zeros((N,), dtype=np.int64)

    return w, t, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I, delayed_spike
