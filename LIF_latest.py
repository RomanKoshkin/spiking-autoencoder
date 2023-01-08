import time
import numpy as np
from numba import jit, prange
from numba.typed import List
import matplotlib.pyplot as plt
from LIFconstants import *
from LIFhelpers import *
from tqdm import trange

np.random.seed(seed)


# @jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def sim(steps, t, w, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I, delayed_spike):
    global N, NE, NI, sigJ, JEEinit, Jmin, Jmax, cEE, cII, cEI, cIE, JEE, JII, JEI, JIE
    global dt, dt
    global refractory_period, V_E, V_I, EL, Vth, Vr, Vspike, U, taustf, taustd
    global tau_ampa, tau_nmda, tau_gaba, TAU_EXCITATORY, TAU_INHIBITORY
    global EPSILON, float_type

    VV = np.ones((N, steps), dtype=float_type)

    for i in range(steps):

        # parallelize over the cores:
        for ii in prange(N):

            if AP[ii] == 1:
                if (t > 1):
                    in_refractory[ii] = refractory_period + np.random.rand() * 0.1
                else:
                    in_refractory[ii] = refractory_period

                AP[ii] = 0

            if np.abs(in_refractory[ii]) < EPSILON:
                delayed_spike[ii] = 1
            else:
                delayed_spike[ii] = 0

            # reset the currents (we recalculate from scratch for each neuron)
            I_E[ii] = 0.0
            I_I[ii] = 0.0

            for jj in range(N):
                bump = neur_type_mask[jj] * F[jj] * D[jj] * delayed_spike[jj] * w[ii, jj]
                antibump = (1.0 - neur_type_mask[jj]) * F[jj] * D[jj] * delayed_spike[jj] * w[ii, jj]

                ampa[ii, jj] += (-ampa[ii, jj] / tau_ampa + bump) * dt
                nmda[ii, jj] += (-nmda[ii, jj] / tau_nmda + bump) * dt
                gaba[ii, jj] += (-gaba[ii, jj] / tau_gaba + antibump) * dt

                I_E[ii] += -ampa[ii, jj] * (V[ii] - V_E) - 0.1 * nmda[ii, jj] * (V[ii] - V_E) + 10
                I_I[ii] += -gaba[ii, jj] * (V[ii] - V_I)

            dV[ii] = (-(V[ii] - EL) / tau[ii] + I_E[ii] + I_I[ii]) * dt

            if V[ii] >= Vspike:
                V[ii] = Vr

            if in_refractory[ii] > EPSILON:  # FIXME: ?????
                dV[ii] = 0.0

            V[ii] += dV[ii]

            if V[ii] > Vth:
                V[ii] = Vspike
                AP[ii] = 1

                # STPonSpike
                F[ii] += U * (1.0 - F[ii])
                D[ii] -= D[ii] * F[ii]

            # STPonNoSpike
            F[ii] += dt * (U - F[ii]) / taustf
            D[ii] += dt * (1.0 - D[ii]) / taustd

            in_refractory[ii] -= dt
            VV[ii, i] = V[ii]

        t += dt

    return w, t, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I, delayed_spike, VV


w, t, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I, delayed_spike = init()
steps = 100
NID, T = [], []

vvv = []

for i in trange(1000):
    (w, t, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I, delayed_spike,
     VV) = sim(steps, t, w, AP, neur_type_mask, tau, V, ampa, nmda, gaba, in_refractory, F, D, dV, I_E, I_I,
               delayed_spike)

    vvv += VV[0, :].tolist()
    times = np.arange(t - 1 + dt, t + dt, dt)
    nid, tid = np.where(VV > Vth)
    NID += nid.tolist()
    T += times[tid].tolist()