alpha = 50.0  # Degree of log-STDP (50.0)
JEI = 0.15  # 0.15 or 0.20
T = 1800000  # simulation time ms
h = 0.01  # time step ms ??????
cEE = 0.4  # probability of connection EE
cIE = 0.2  # probability of connection IE
cEI = 0.5  # probability of connection EI
cII = 0.5  # probability of connection II
JEE = 0.45  # Synaptic weights ????????????
JEEinit = 0.46  # ?????????????
JIE = 0.15  #
JII = 0.06  #
JEEh = 0.15  # initial conditions of synaptic weights. Standard synaptic weight E-E
sigJ = 0.3  #
Jtmax = 0.25  # J_maxˆtot
Jtmin = 0.01  # J_minˆtot # ??? NOT IN THE PAPER
hE = 1.0  # Threshold of update of excitatory neurons
hI = 1.0  # Threshold of update of inhibotory neurons
IEex = 2.0  # Amplitude of steady external input to excitatory neurons
IIex = 0.5  # Amplitude of steady external input to inhibitory neurons
mex = 0.3  # mean of external input
sigex = 0.1  # variance of external input
tmE = 2.5  # Average intervals of update ms = t_Eud EXCITATORY decreased by a factor of 5
tmI = 2.5  # Average intervals of update ms = t_Iud INHIBITORY decreased by a factor of 5
trec = 600.0  # STD = recovery time constant (tau_sd p.13 and p.12)
Jepsilon = 0.001  #
tpp = 15.1549  # STDP: tau_p <<<<<<<<<
tpd = 120.4221  # STDP: tau_d <<<<<<<<<
twnd = 500.0  # STDP = window lenght ms
g = 2.5  # ??????
itauh = 100  # homeostatic = decay time of homeostatic plasticity (100s)
hsd = 0.1  # ??
hh = 10.0  # ??????? SOME MYSTERIOUS PARAMETER
Ip = 1.0  # External current applied to randomly chosen excitatory neurons
a = 0.20  # Fraction of neurons to which this external current is applied
xEinit = 0.02  # the probability that an excitatory neurons spikes at the beginning of the simulation
xIinit = 0.01  # the probability that an inhibitory neurons spikes at the beginning of the simulation
tinit = 100.00  # period of time after which STDP kicks in (100.0)
U = 0.8  # ???? <<<<<<<<<
taustf = 250.0  # STP = STF time constant <<<<<<<<<
taustd = 350.0  # STP = STD time constant <<<<<<<<<
Cp = 0.14  # ??
Cd = 0.02  #  ??
HAGA = True
symmetric = True

Jmax = 5.0 * JEE
Jmin = 0.01 * JEE

# these are not passed to 'model.setParams()', but are passed to the model constructor:
NEo = 0
NE = 400
NM = 0  # out of NE excitatory neurons, NM are motor neurons
NI = 80
N = NE + NI
dt = 0.01
case = 3
wide = True

# environment parameters:
gauss = True
sigma = 0.5
decision_margin = 0.01
gridsize = 20
speed = 1
paddle_len = 6
paddle_width = 1
restart = True

refractory_period = 1

# equilibrium potentials:
V_E = 0.0
V_I = -80.0  # equilibrium potential for the inhibitory synapse
EL = -65.0  # leakage potential, mV

# critical voltages:
Vth = -55.0  # threshold after which an AP is fired,   mV
Vr = -70.0  # reset voltage (after an AP is fired), mV
Vspike = 10.0

U = 0.98
taustf = 50.0
taustd = 250.0

# taus
tau_ampa = 8
tau_nmda = 100
tau_gaba = 8

# membrane taus
TAU_EXCITATORY = 10
TAU_INHIBITORY = 20

# random
seed = 20
EPSILON = 0.001