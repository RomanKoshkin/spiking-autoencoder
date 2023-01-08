#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

// #include "model.h"

using namespace std;

// double Model::dice(){
// 	return rand()/(RAND_MAX + 1.0);
// }

void Model::saveRecentSpikes(int i, double t) {
    sphist[i].push_back(t);
    // remove spikes older than DEQUE_T_LEN
    if ((t - sphist[i].front()) > DEQUE_LEN_MS) {
        sphist[i].pop_front();
    }
}

double Model::dice() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(m_mt);
}

double Model::ngn() {
    // sample from a normal distribution based on two uniform distributions
    // WHY IS IT SO?
    double u = Model::dice();
    double v = Model::dice();
    return sqrt(-2.0 * log(u)) * cos(2.0 * pi * v);
}

// choose the neuron ids that will be updated at the current time step
vector<int> Model::rnd_sample(int ktmp, int Ntmp) {  // when ktmp << Ntmp
    vector<int> smpld;
    int xtmp;
    bool tof;
    while (smpld.size() < ktmp) {
        xtmp = (int)floor(Ntmp * Model::dice());
        tof = true;
        // make sure that the sampled id isn't the same as any of the previous
        // ones
        for (int i = 0; i < smpld.size(); i++) {
            if (xtmp == smpld[i]) {
                tof = false;
            }
        }
        if (tof) smpld.push_back(xtmp);
    }
    return smpld;
}

double Model::fd(double x, double alpha) { return log(1.0 + alpha * x) / log(1.0 + alpha); }

void Model::STPonSpike(int i) {
    F[i] += UU[i] * (1 - F[i]);  // U = 0.6
    D[i] -= D[i] * F[i];

    // remove it from the set of spiking neurons
    it = spts.find(i);
    if (it != spts.end()) {
        spts.erase(it++);
    }
    // and turn it OFF
    x[i] = 0;
}

void Model::STPonNoSpike() {
    // EVERY 10 th timestep becuase STD is slow
    // check it yourself:
    // for i in np.arange(0, 2, 0.01):
    //	  print(np.round(i , 2), int(np.floor(i / 0.01)) % 10 == 0)
    // hsd explained
    // in the left-hand side of the a diff equation we have dy/dt
    // so to get dy, we need to multiply the right-hand side by dt in ms (the
    // units of taustd/taustf), the time step in this simulation is 0.01 ms, so
    // our dt (hsd in this case) would be 0.01 if we updated dy every time step
    // (that is 0.01 ms). But we update every 10th timestep, so we use 0.1
    // instead of 0.01
    if (((int)floor(t / h)) % 10 == 0) {
        for (int i = 0; i < NE; i++) {
            F[i] += hsd * (UU[i] - F[i]) / taustf;  // @@ don't forget about hsd!!!
            D[i] += hsd * (1.0 - D[i]) / taustd;

            // if (t < 2000000) {
            // 	F[i] += hsd * (UU[i] - F[i]) / taustf; // @@ don't forget about
            // hsd!!! 	D[i] += hsd * (1.0 - D[i]) / taustd; } else {
            // 	// hsd = 1.0;
            // //
            // @@ don't forget about hsd!!! 	k1 = (UU[i] - F[i]) / taustf; k2
            // = (UU[i] - (F[i] + 0.5 * hsd * k1)) / taustf; // @@ don't forget
            // about hsd!!! 	k3 = (UU[i]
            // - (F[i] + 0.5 * hsd * k2)) / taustf; 	k4 = (UU[i] - (F[i] +
            // hsd
            // *
            // k3)) / taustf; 	F[i] += hsd * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            // / 6.0;

            // 	k1 = (1.0 - D[i]) / taustd;
            // 	k2 = (1.0 - (D[i] + 0.5 * hsd * k1)) / taustd;
            // 	k3 = (1.0 - (D[i] + 0.5 * hsd * k2)) / taustd;
            // 	k4 = (1.0 - (D[i] + hsd * k3)) / taustd;
            // 	D[i] += hsd * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            // }
        }
    }
}

void Model::saveDSPTS() {
    ofstream ofsDSPTS;
    // ofsDSPTS.open("DSPTS_" + std::to_string(cell_id) + "_" +
    // std::to_string(t));
    ofsDSPTS.open("data/DSPTS_" + std::to_string(cell_id));
    ofsDSPTS.precision(10);

    for (int i = 0; i < NE; i++) {
        ofsDSPTS << i;
        for (int sidx = 0; sidx < dspts[i].size(); sidx++) {
            ofsDSPTS << " " << dspts[i][sidx];
        }
        ofsDSPTS << endl;
    }
}

void Model::saveX() {
    ofstream ofsX;
    // ofsX.open("X_" + std::to_string(cell_id) + "_" + std::to_string(t));
    ofsX.open("data/X_" + std::to_string(cell_id));
    ofsX.precision(10);
    ofsX << x[0];
    for (int i = 1; i < N; i++) {
        ofsX << " " << x[i];
    }
    ofsX << endl;
}

void Model::loadDSPTS(string tt) {
    dspts.clear();
    deque<double> iideque;
    // ifstream file("DSPTS_" + std::to_string(cell_id) + "_" + tt);
    ifstream file("data/DSPTS_" + std::to_string(cell_id));

    if (!file.is_open()) {
        cout << "DSPTS file not found." << endl;
        throw "DSPTS file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString(line.c_str());
        iideque.pop_front();
        dspts.push_back(iideque);
    }
    file.close();
    cout << "DSPTS loaded" << endl;
}

void Model::loadX(string tt) {
    x.clear();
    deque<double> iideque;
    ifstream file("data/X_" + std::to_string(cell_id));
    if (!file.is_open()) {
        cout << "X file not found." << endl;
        throw "X file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString(line.c_str());
        for (int i = 0; i < N; i++) {
            x.push_back(iideque[i]);
        }
    }
    file.close();
    cout << "X loaded" << endl;
}

deque<double> Model::SplitString(string line) {
    deque<double> iideque;
    string temp = "";
    for (int i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
            iideque.push_back(stod(temp));
            temp = "";
        } else {
            temp.push_back(line[i]);
        }
    }
    iideque.push_back(stod(temp));
    return iideque;
}

deque<int> Model::SplitString_int(string line) {
    deque<int> iideque;
    string temp = "";
    for (int i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
            iideque.push_back(stoi(temp));
            temp = "";
        } else {
            temp.push_back(line[i]);
        }
    }
    return iideque;
}

void Model::updateMembranePot(int i) {
    // WE update the membrane potential of the chosen POSTsynaptic excitatory neuron i (eq.4,
    // p.12) -threshold of update + steady exc input * mean/var of external stim
    // input

    // first inject random "driving" noise
    if ((i >= NE - NEo) && (i < NE)) {
        u = -hE + IEex * (mex + sigex * ngn()) * 0.6;  // pentagon, p.12
    } else {
        u = -hE + IEex * (mex + sigex * ngn());  // pentagon, p.12
    }

    // in progress on total inhibition
    double totalInhibition = 0.0;
    if (inhibition_mode == 1) {
        for (int ii = NE; ii < NE + NI; ii++) {
            totalInhibition += getRecent(ii);
        }
    }

    // then we go over all PREsynaptic neurons that are spiking now
    for (const auto& j : spts) {
        // if a PREsynaptic spiking neuron happens to be excitatory,
        if (j < NE) {
            u += F[j] * D[j] * Jo[i][j];
            // if a PREsynaptic spiking neuron happens to be inhibitory,
            Uexc[i].back() += F[j] * D[j] * Jo[i][j];
        } else {
            if (inhibition_mode == 0) {
                u += Jo[i][j];  // we add because the E<-I weights are negative
                Uinh[i].back() += Jo[i][j];
            }
        }
    }
    if (inhibition_mode == 1) {
        u -= totalInhibW * totalInhibition;
        Uinh[i].back() = -totalInhibW * totalInhibition;
    }
}

double Model::getRecent(int i) {
    int J = sphist[i].size();  // J is the number if presynaptic spikes
    // we have an array of pointers
    double acc = 0.0;  // zero accumulator
    for (int j = 0; j < J; j++) {
        // the weight of the spike will be the (exponentially) lower the older it is
        float expw = exp(0.08 * ((sphist[i][j]) - (t)));
        acc += expw;
    }
    return acc;
}

void Model::checkIfStim(int i) {
    if (hStim[i] == 1) {
        if (dice() < stimIntensity[i]) {
            u += Ip;
        }
        // std::cout << "stim @ t=" << t << " on neuron " << i << std::endl;
    }
}

double Model::heaviside(double x) { return double(x > 0); }

double Model::alpha_function_LTP(double wgt) {
    return exp(-wgt / alpha_tau + 1.0) * heaviside(wgt) * (wgt / alpha_tau);
}

double Model::alpha_function_LTD(double wgt) {
    return exp((wgt - 1) / alpha_tau + 1.0) * heaviside(wgt + 1.0) * ((1.0 - wgt) / alpha_tau);
}

double Model::tanh_LTP(double wgt) { return -tanh(30 * (wgt - Jmax)); }

double Model::tanh_LTD(double wgt) { return tanh(30 * wgt); }

void Model::STDP(int i) {
    /* First (LTD), we treat the chosen neuron as PREsynaptic and loop over all
      the POSTSYNAPTIC excitatory neurons that THE CHOSEN NEURON synapses on.
      Since we're at time t (and this is the latest time), the spikes recorded on
      those "POSTsynaptic" neurons will have an earlier timing than the spike
      recorded on the currently chosen neuron (that we treat as PREsynaptic). This
      indicates that the synaptic weight between this chosen neuron (presynaptic)
      and all the other neurons (postsynaptic) will decrease.  */

    for (int ip = 0; ip < NE; ip++) {
        if (Jo[ip][i] > Jepsilon && t > tinit) {
            // dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
            for (const auto& tt : dspts[ip]) {
                if (HAGA == 1) {
                    // STP-dependent (new HAGA)
                    // double alphaLTP = alpha_function_LTP(Jo[ip][i]); // <<<<
                    // ALPHA <<<<<<<<<<<<<<<<<<<<<<<<< double alphaLTD =
                    // alpha_function_LTD(Jo[ip][i]); // <<<< ALPHA
                    // <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTP = tanh_LTP(Jo[ip][i]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTD = tanh_LTD(Jo[ip][i]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    Jo[ip][i] += F[i] * D[i] * alphaLTP *
                                 (Cp * exp((tt - t) / tpp) -
                                  alphaLTD * fd(Jo[ip][i] / JEE, alpha) * Cd * exp((tt - t) / tpd));
                } else {
                    // STP-independent (new Hiratani)
                    Jo[ip][i] += (Cp * exp((tt - t) / tpp) -
                                  fd(Jo[ip][i] / JEE, alpha) * Cd * exp((tt - t) / tpd));
                }
            }
            // we force the weights to be no less than Jmin, THIS WAS NOT IN THE
            // PAPER
            if (Jo[ip][i] > Jmax) Jo[ip][i] = Jmax;
            if (Jo[ip][i] < Jmin) Jo[ip][i] = Jmin;
        }
    }

    // (LTP)

    /* Jinidx is a list of lists (shape (2500 POSTsyn, n PREsyn)). E.g. if in
      row 15 we have number 10, it means that the weight between POSTsynaptic
      neuron 15 and presynaptc neuron 10 is greater than Jepsilon */

    /* we treat the currently chosen neuron as POSTtsynaptic, and we loop over
      all the presynaptic neurons that synapse on the current postsynaptic neuron.
      At time t (the latest time, and we don't yet know any spikes that will
      happen in the future) all the spikes on the presynaptic neurons with id j
      will have an earlier timing that the spike on the currently chosen neuron i
      (that we treat as postsynaptic for now). This indicates that the weights
      between the chosen neuron treated as post- synaptic for now and all the
      other neurons (treated as presynaptic for now) will be potentiated.  */

    for (const auto& j : Jinidx[i]) {
        // at each loop we get the id of the jth presynaptic neuron with J >
        // Jepsilon
        if (t > tinit) {
            for (const auto& tt : dspts[j]) {
                // we loop over all the spike times on the jth PRESYNAPTIC
                // neuron
                if (HAGA == 1) {
                    // double alphaLTP = alpha_function_LTP(Jo[i][j]); // <<<<
                    // ALPHA <<<<<<<<<<<<<<<<<<<<<<<<< double alphaLTD =
                    // alpha_function_LTD(Jo[i][j]); // <<<< ALPHA
                    // <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTP = tanh_LTP(Jo[i][j]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTD = tanh_LTD(Jo[i][j]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    // STP-dependent (по новому определению Haga)
                    Jo[i][j] += F[j] * D[j] * alphaLTP *
                                (Cp * exp(-(t - tt) / tpp) -
                                 alphaLTD * fd(Jo[i][j] / JEE, alpha) * Cd * exp(-(t - tt) / tpd));
                } else {
                    // STP-independent (по новому определению Hiratani: не так
                    // как было у Хиратани в статье, а просто без F*D)
                    Jo[i][j] += (Cp * exp(-(t - tt) / tpp) -
                                 fd(Jo[i][j] / JEE, alpha) * Cd * exp(-(t - tt) / tpd));
                }
            }
            // we force the weights to be no more than Jmax, THIS WAS NOT IN THE
            // PAPER
            if (Jo[i][j] > Jmax) Jo[i][j] = Jmax;
            if (Jo[i][j] < Jmin) Jo[i][j] = Jmin;
        }
    }
}

void Model::symSTDP(int i) {
    /* First (LTD), we treat the chosen neuron as PREsynaptic and loop over all
      the POSTSYNAPTIC excitatory neurons that THE CHOSEN NEURON synapses on.
      Since we're at time t (and this is the latest time), the spikes recorded on
      those "POSTsynaptic" neurons will have an earlier timing than the spike
      recorded on the currently chosen neuron (that we treat as PREsynaptic). This
      indicates that the synaptic weight between this chosen neuron (presynaptic)
      and all the other neurons (postsynaptic) will decrease.  */

    for (int ip = 0; ip < NE; ip++) {
        if (Jo[ip][i] > Jepsilon && t > tinit) {
            // dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
            for (const auto& tt : dspts[ip]) {
                if (HAGA == 1) {
                    // STP-dependent (new HAGA)
                    // double alphaLTP = alpha_function_LTP(Jo[ip][i]); // <<<<
                    // ALPHA <<<<<<<<<<<<<<<<<<<<<<<<< double alphaLTD =
                    // alpha_function_LTD(Jo[ip][i]); // <<<< ALPHA
                    // <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTD = tanh_LTD(Jo[ip][i]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    Jo[ip][i] -= F[i] * D[i] * alphaLTD * Cd * exp((tt - t) / tpp);
                } else {
                    // STP-independent (new Hiratani)
                    Jo[ip][i] -= Cd * exp((tt - t) / tpp);
                }
            }
            // we force the weights to be no less than Jmin, THIS WAS NOT IN THE
            // PAPER
            if (Jo[ip][i] > Jmax) Jo[ip][i] = Jmax;
            if (Jo[ip][i] < Jmin) Jo[ip][i] = Jmin;
        }
    }

    // (LTP)

    /* Jinidx is a list of lists (shape (2500 POSTsyn, n PREsyn)). E.g. if in
      row 15 we have number 10, it means that the weight between POSTsynaptic
      neuron 15 and presynaptc neuron 10 is greater than Jepsilon */

    /* we treat the currently chosen neuron as POSTtsynaptic, and we loop over
      all the presynaptic neurons that synapse on the current postsynaptic neuron.
      At time t (the latest time, and we don't yet know any spikes that will
      happen in the future) all the spikes on the presynaptic neurons with id j
      will have an earlier timing that the spike on the currently chosen neuron i
      (that we treat as postsynaptic for now). This indicates that the weights
      between the chosen neuron treated as post- synaptic for now and all the
      other neurons (treated as presynaptic for now) will be potentiated.  */

    for (const auto& j : Jinidx[i]) {
        // at each loop we get the id of the jth presynaptic neuron with J >
        // Jepsilon
        if (t > tinit) {
            for (const auto& tt : dspts[j]) {
                // we loop over all the spike times on the jth PRESYNAPTIC
                // neuron
                if (HAGA == 1) {
                    // double alphaLTP = alpha_function_LTP(Jo[i][j]); // <<<<
                    // ALPHA <<<<<<<<<<<<<<<<<<<<<<<<< double alphaLTD =
                    // alpha_function_LTD(Jo[i][j]); // <<<< ALPHA
                    // <<<<<<<<<<<<<<<<<<<<<<<<<
                    double alphaLTP = tanh_LTP(Jo[i][j]);  // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
                    // STP-dependent (по новому определению Haga)
                    Jo[i][j] += F[j] * D[j] * alphaLTP * Cp * exp(-(t - tt) / tpp);
                } else {
                    // STP-independent (по новому определению Hiratani: не так
                    // как было у Хиратани в статье, а просто без F*D)
                    Jo[i][j] += Cp * exp(-(t - tt) / tpp);
                }
            }
            // we force the weights to be no more than Jmax, THIS WAS NOT IN THE
            // PAPER
            if (Jo[i][j] > Jmax) Jo[i][j] = Jmax;
            if (Jo[i][j] < Jmin) Jo[i][j] = Jmin;
        }
    }
}

// initialize the weight matrix
vector<vector<double>> Model::calc_J(double JEEinit, double JEI) {
    vector<vector<double>> J;
    int mcount = 0;
    for (int i = 0; i < NE; i++) {
        J.push_back(dvec);
        for (int j = 0; j < NE; j++) {
            J[i].push_back(0.0);
            // first E-E weights consistent with the E-E connection probability

            if (i != j && dice() < cEE) {
                // @@ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                J[i][j] = JEEinit * (1.0 + sigJ * ngn());

                // if some weight is out of range, we clip it
                if (J[i][j] < Jmin) J[i][j] = Jmin;
                if (J[i][j] > Jmax) J[i][j] = Jmax;
            }
        }
        // then the E-I weights
        for (int j = NE; j < N; j++) {
            J[i].push_back(0.0);  // here the matrix J is at first of size 2500,
                                  // we extend it
            if (dice() < cEI) {
                J[i][j] -= JEI; /* becuase jth presynaptic inhibitory synapsing
                        on an ith excitatory postsynaptic neuron should inhibit it.
                        Hence the minus */
            }
        }
    }

    // then the I-E and I-I weights
    for (int i = NE; i < N; i++) {
        J.push_back(dvec);
        for (int j = 0; j < NE; j++) {
            J[i].push_back(0.0);
            if (dice() < cIE) {
                J[i][j] += JIE;
            }
        }
        for (int j = NE; j < N; j++) {
            J[i].push_back(0.0);
            if (i != j && dice() < cII) {
                J[i][j] -= JII;
            }
        }
    }

    // prohibit certain weights
    for (int i = NE - NEo; i < N; i++) {  // <<<<<<<<<<
                                          // NE-NEo or N
        for (int j = NE - NEo; j < NE; j++) {
            J[i][j] = 0.0;
        }
    }
    for (int i = NE - NEo; i < NE; i++) {
        for (int j = NE - NEo; j < N; j++) {
            J[i][j] = 0.0;
        }
    }

    return J;
}

void Model::reinitFD() {
    for (int i = 0; i < NE; i++) {
        F[i] = U;
        D[i] = 1.0;
        UU[i] = U;
    }
}

void Model::reinit_Jinidx() {
    Jinidx.clear();
    for (int i = 0; i < NE; i++) {
        Jinidx.push_back(ivec);  // shape = (3000, max3000)
        for (int i2 = 0; i2 < NE; i2++) {
            if (Jo[i][i2] > Jepsilon) {
                Jinidx[i].push_back(i2);
            }
        }
    }
}

void Model::saveSpts() {
    //   for (it = spts.begin(); it != spts.end(); it++) {
    //       cout << *it << endl;
    //   }
    // same as above (using an iterator, but more pythonic for ... in...)
    ofstream ofsSpts;
    // ofsSpts.open("SPTS_" + to_string(cell_id) + "_" + to_string(t));
    ofsSpts.open("data/SPTS_" + to_string(cell_id));

    for (int spt : spts) {
        ofsSpts << spt << " ";
    }
}

void Model::loadSpts(string tt) {
    spts.clear();

    deque<int> iideque;
    // ifstream file("SPTS_" + to_string(cell_id) + "_" + tt);
    ifstream file("data/SPTS_" + to_string(cell_id));

    if (!file.is_open()) {
        cout << "SPTS file not found." << endl;
        throw "SPTS file not found.";
    }
    string line;
    while (getline(file, line)) {
        iideque = SplitString_int(line.c_str());
        for (auto spt : iideque) {
            spts.insert(spt);
        }
    }
    file.close();
    cout << "SPTS loaded" << endl;
}

void Model::setParams(Model::ParamsStructType params) {
    alpha = params.alpha;        // Degree of log-STDP (50.0)
    JEI = params.JEI;            // 0.15 or 0.20
    T = params.T;                // simulation time, ms (1800*1000.0)
    h = params.h;                // time step, ms ??????
    cEE = params.cEE;            // 0.2
    cIE = params.cIE;            // 0.2
    cEI = params.cEI;            // 0.5
    cII = params.cII;            // 0.5
    JEE = params.JEE;            // 0.15
    JEEinit = params.JEEinit;    // 0.18
    JIE = params.JIE;            // 0.15
    JII = params.JII;            // 0.06
    JEEh = params.JEEh;          // Standard synaptic weight E-E 0.15
    sigJ = params.sigJ;          // 0.3
    Jtmax = params.Jtmax;        // J_maxˆtot (0.25)
    Jtmin = params.Jtmin;        // J_minˆtot // ??? NOT IN THE PAPER (0.01)
    hE = params.hE;              // Threshold of update of excitatory neurons 1.0
    hI = params.hI;              // Threshold of update of inhibotory neurons 1.0
    IEex = params.IEex;          // Amplitude of steady external input to excitatory
                                 // neurons 2.0
    IIex = params.IIex;          // Amplitude of steady external input to inhibitory
                                 // neurons 0.5
    mex = params.mex;            // mean of external input 0.3
    sigex = params.sigex;        // variance of external input 0.1
    tmE = params.tmE;            // t_Eud EXCITATORY 5.0
    tmI = params.tmI;            // t_Iud INHIBITORY 2.5
    trec = params.trec;          // recovery time constant (tau_sd, p.13 and p.12) 600.0
    Jepsilon = params.Jepsilon;  // ???????? 0.001
    tpp = params.tpp;            // tau_p 20.0
    tpd = params.tpd;            // tau_d 40.0
    Cp = params.Cp;
    Cd = params.Cd;
    twnd = params.twnd;      // STDP window lenght, ms 500.0
    g = params.g;            // ???? 1.25
    itauh = params.itauh;    // decay time of homeostatic plasticity,s (100)
    hsd = params.hsd;        // integration time step (10 * simulation timestep)
    hh = params.hh;          // SOME MYSTERIOUS PARAMETER 10.0
    Ip = params.Ip;          // External current applied to randomly chosen excitatory
                             // neurons 1.0
    a = params.a;            // Fraction of neurons to which this external current is
                             // applied 0.20
    xEinit = params.xEinit;  // prob that an exc neurons spikes at the beginning
                             // of the simulation 0.02
    xIinit = params.xIinit;  // prob that an inh neurons spikes at the beginning
                             // of the simulation 0.01
    tinit = params.tinit;    // period of time after which STDP kicks in 100.0

    // recalculate values that depend on the parameters
    SNE = (int)floor(NE * h / tmE + 0.001);
    SNI = max(1, (int)floor(NI * h / tmI + 0.001));

    Jmax = 5.0 * JEE;   // ???
    Jmin = 0.01 * JEE;  // ????
    // Cp = 0.1*JEE;              // must be 0.01875 (in the paper)
    // Cd = Cp*tpp/tpd;           // must be 0.0075 (in the paper)
    hsig = 0.001 * JEE;               // i.e. 0.00015 per time step (10 ms)
    NEa = (int)floor(NE * a + 0.01);  // Exact number of excitatory neurons stimulated externally
    pmax = NE / NEa;

    tauh = itauh * 1000.0;  // decay time of homeostatic plasticity, in ms

    U = params.U;

    taustf = params.taustf;
    taustd = params.taustd;
    HAGA = params.HAGA;
    symmetric = params.symmetric;

    reinitFD();
}

Model::retParamsStructType Model::getState() {
    Model::retParamsStructType ret_struct;
    ret_struct.alpha = alpha;
    ret_struct.JEI = JEI;
    ret_struct.T = T;
    ret_struct.h = h;
    ret_struct.NE = NE;
    ret_struct.NI = NI;
    ret_struct.cEE = cEE;
    ret_struct.cIE = cIE;
    ret_struct.cEI = cEI;
    ret_struct.cII = cII;
    ret_struct.JEE = JEE;
    ret_struct.JEEinit = JEEinit;
    ret_struct.JIE = JIE;
    ret_struct.JII = JII;
    ret_struct.JEEh = JEEh;
    ret_struct.sigJ = sigJ;
    ret_struct.Jtmax = Jtmax;
    ret_struct.Jtmin = Jtmin;
    ret_struct.hE = hE;
    ret_struct.hI = hI;
    ret_struct.IEex = IEex;
    ret_struct.IIex = IIex;
    ret_struct.mex = mex;
    ret_struct.sigex = sigex;
    ret_struct.tmE = tmE;
    ret_struct.tmI = tmI;
    ret_struct.trec = trec;
    ret_struct.Jepsilon = Jepsilon;
    ret_struct.tpp = tpp;
    ret_struct.tpd = tpd;
    ret_struct.twnd = twnd;
    ret_struct.g = g;
    ret_struct.itauh = itauh;
    ret_struct.hsd = hsd;
    ret_struct.hh = hh;
    ret_struct.Ip = Ip;
    ret_struct.a = a;
    ret_struct.xEinit = xEinit;
    ret_struct.xIinit = xIinit;
    ret_struct.tinit = tinit;

    ret_struct.Jmin = Jmin;
    ret_struct.Jmax = Jmax;
    ret_struct.Cp = Cp;
    ret_struct.Cd = Cd;
    ret_struct.SNE = SNE;
    ret_struct.SNI = SNI;
    ret_struct.NEa = NEa;
    ret_struct.t = t;

    ret_struct.U = U;
    ret_struct.taustf = taustf;
    ret_struct.taustd = taustd;
    ret_struct.HAGA = HAGA;
    ret_struct.symmetric = symmetric;

    return ret_struct;
}

void Model::initLIF() {
    for (int i = 0; i < N; i++) {
        if (dice() > 0.5) {
            AP.push_back(0);
        } else {
            AP.push_back(1);
        }
        V.push_back(EL);
        in_refractory.push_back(0.0);
        dV.push_back(0.0);
        I_E.push_back(0.0);
        I_I.push_back(0.0);
        delayed_spike.push_back(0);
        if (i < NE) {
            neur_type_mask.push_back(0.0);
            tau.push_back(TAU_EXCITATORY);
        } else {
            neur_type_mask.push_back(1.0);
            tau.push_back(TAU_INHIBITORY);
        }
        ampa.push_back(dvec);
        nmda.push_back(dvec);
        gaba.push_back(dvec);
        for (int j = 0; j < N; j++) {
            ampa[i].push_back(0.0);
            nmda[i].push_back(0.0);
            gaba[i].push_back(0.0);
        }
    }
}