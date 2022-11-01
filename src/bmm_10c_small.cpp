// Reduced binary model of short- and long-term synaptic plasticity
//
// Original code by Created by Naoki Hiratani (N.Hiratani@gmail.com)
// Comments and Python-friendly interactive implementation
// by Roman Koshkin (roman.koshkin@gmail.com)

// добавил alphaLTP, alphaLTD к функции STDP

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <boost/format.hpp>
#include <chrono>
#include <cmath>
#include <deque>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "model_methods.h"
#include "utils.h"

using namespace std;

// to complile
// g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx
// bmm_9_haga_grid.cpp -o ./bmm.dylib g++ -std=gnu++11 -O3 -dynamiclib
// -ftree-vectorize -march=native -mavx bmm_7_haga.cpp -o ./bmm.dylib sudo
// /usr/bin/g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native
// -mavx bmm_8_haga.cpp -o ./bmm.dylib icc -std=gnu++11 -O3 -shared -fPIC
// bmm_5_haga.cpp -o ./bmm.dylib g++ -std=gnu++11 -Ofast -shared -fPIC
// -ftree-vectorize -march=native -mavx -fopenmp bmm_9_haga.cpp -o ./bmm.dylib

void Model::sim(int interval) {
    // std::cout << "SNE = " << SNE << ", SNI = " << SNI << std::endl;
    // std::cout << "t = " << t << std::endl;
    Timer timer;

    // HAVING INITIALIZED THE NETWORK, WE go time step by time step
    while (interval > 0) {
        // // save STP states every millisecond
        // if( (int)floor(t/h) % 1000 == 0) {
        //     saveSTP();
        // }

        t += h;
        interval -= 1;

        // we decide which EXCITATORY neurons will be updated
        // they may or may not be spiking at the current step
        smpld = rnd_sample(SNE, NE);

        // to log the distribution of sampled neurons
        // ofsb << smpld[0] << " " << smpld[1] << " " << smpld[2] << " " <<
        // smpld[3] << " " << smpld[4] << " " << std::endl;

        // we cycle through those chosen neurons
        for (const int& i : smpld) {
            // STP (empty square, eq. 6 p.12)
            // if a chosen neuron is ALREADY on
            if (x[i] == 1) {
                STPonSpike(i);
            }
            // either way
            updateMembranePot(i);
            checkIfStim(i);

            // WE RECORD A SPIKE and PERFORM AN STDP on the chosen neuron if it
            // spikes u > 0
            if (u > theta[i]) {
                // if the POSTsynaptic neuron chosen for update exceeds the
                // threshold, we save its ID in the set "spts"
                spts.insert(i);

                /* dspts saves the TIME of this spike to a DEQUE, such that each
                        row id of this deque corresponds to the id of the spiking
                        neuron. The row records the times at which that neuron emitted
                        at spike. */
                dspts[i].push_back(t);  // SHAPE: (n_postsyn x pytsyn_sp_times)
                x[i] = 1;
                // // record a spike on an EXCITATORY neuron (because STDP is
                // only on excitatory neurons)
                if (saveflag == 1) {
                    ofsr << t << " " << i << endl;  // record a line to file
                    saveRecentSpikes(i, t);
                }

                if (STDPon) {
                    if (symmetric) {
                        STDP(i);
                    } else {
                        symSTDP(i);
                    }
                }

                if (i > NE - NEo) {
                    theta[i] += 0.03;
                }
            }
        }

        // exponentially decaying threshold for excitatory neurons
        for (int i_ = NE - NEo; i_ < NE; i_++) {
            theta[i_] *= 0.999;
        }

        // we sample INHIBITORY neurons to be updated at the current step
        smpld = rnd_sample(SNI, NI);

        for (const int i_ : smpld) {
            int i = NE + i_;
            /* if this inhibitory neuron is spiking we set it to zero
                  in the binary vector x and remove its index from the set
                  of currently spiking neurons */

            // crazy optimization, but in fact eq.(5) p.12 (filled circle)
            if (x[i] == 1) {
                it = spts.find(i);
                if (it != spts.end()) {
                    // removing a spike time from the SET of spikes on
                    // inhibitory neurons is the same as subtracting them
                    spts.erase(it++);
                }
                x[i] = 0;
            }

            // update the membrane potential on a chosen inhibitory neuron
            u = -hI + IIex * (mex + sigex * ngn());  // hexagon, eq.5, p.12

            for (const int& j : spts) {
                u += Jo[i][j];
            }

            // if the membrane potential on the currently chosen INHIBITORy
            // neuron is greater than the threshold, we record a spike on this
            // neuron.
            if (u > 0) {
                spts.insert(i);
                x[i] = 1;
                // // record a spike on an INHIBITORY NEURON
                if (saveflag == 1) {
                    ofsr << t << " " << i << endl;
                    // we dont keep a history of recent spikes on IHIBITORY spikes
                }
            }
        }

        STPonNoSpike();

        // EVERY 10 ms Homeostatic plasticity, weight clipping, boundary
        // conditions, old spike removal
        if (((int)floor(t / h)) % 1000 == 0) {
            // Homeostatic Depression
            if (homeostatic) {
                for (int i = 0; i < NE; i++) {
                    for (const int& j : Jinidx[i]) {
                        // ?????????? THAT'S NOT EXACTLY WHAT THE PAPER SAYS
                        k1 = (JEEh - Jo[i][j]) / tauh;
                        k2 = (JEEh - (Jo[i][j] + 0.5 * hh * k1)) / tauh;
                        k3 = (JEEh - (Jo[i][j] + 0.5 * hh * k2)) / tauh;
                        k4 = (JEEh - (Jo[i][j] + hh * k3)) / tauh;
                        Jo[i][j] += hh * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 + hsig * ngn();
                        // we clip the weights from below and above
                        if (Jo[i][j] < Jmin)
                            Jo[i][j] = Jmin;  // ????? Jmin is zero, not 0.0015,
                                              // as per Table 1
                        if (Jo[i][j] > Jmax) Jo[i][j] = Jmax;
                    }
                }

                // boundary condition
                for (int i = 0; i < NE; i++) {
                    double Jav = 0.0;
                    for (const int& j : Jinidx[i]) {
                        // find the total weight per each postsynaptic neuron
                        Jav += Jo[i][j];
                    }
                    // find mean weight per each postsynaptic neuron
                    Jav = Jav / ((double)Jinidx[i].size());
                    if (Jav > Jtmax) {
                        for (const int& j : Jinidx[i]) {
                            // if the total weight exceeds Jtmax, we subtract
                            // the excess value
                            Jo[i][j] -= (Jav - Jtmax);
                            // but if a weight is less that Jmin, we set it to
                            // Jmin (clip from below)
                            if (Jo[i][j] < Jmin) {
                                Jo[i][j] = Jmin;
                            }
                        }

                        // if the total weight is less that Jtmin
                    } else if (Jav < Jtmin) {
                        for (const int& j : Jinidx[i]) {
                            /* ???????? we top up each (!!!???) weight by the
                                          difference between the total min and current total
                                          weight */
                            Jo[i][j] += (Jtmin - Jav);
                            // but if a weight is more that Jmax, we clip it to
                            // Jmax
                            if (Jo[i][j] > Jmax) {
                                Jo[i][j] = Jmax;
                            }
                        }
                    }
                }
            }

            // remove spikes older than 500 ms
            for (int i = 0; i < NE; i++) {
                for (int sidx = 0; sidx < dspts[i].size(); sidx++) {
                    // if we have spike times that are occured more than 500 ms
                    // ago, we pop them from the deque
                    if (t - dspts[i][0] > twnd) {
                        dspts[i].pop_front();
                    }
                }
            }
        }

        // EVERY 1s
        if (((int)floor(t / h)) % (1000 * 100) == 0) {
            tidx += 1;  // we count the number of 1s cycles
            int s = 0;
            it = spts.begin();
            while (it != spts.end()) {
                ++s;
                ++it;
            }

            // exit if either no neurons are spiking or too many spiking after t
            // > 200 ms
            if (s == 0 || (s > 1.0 * NE && t > 200.0)) {
                std::cout << "Exiting because either 0 or too many spikes at t =" << t << std::endl;
                break;
            }
        }
    }
}

extern "C" {

// we create a pointer to the object of type Net. The pointer type must be of
// the same type as the object/variable this pointer points to
Model* createModel(int _NE, int _NI, int _NEo, int _cell_id) {
    cout << _NE << " " << _NI << " " << _NEo << " " << _cell_id << " " << endl;
    return new Model(_NE, _NI, _NEo, _cell_id);
}

// NOTE: in progress
double* getRecents(Model* m) {
    const int I = m->NE;  // we only need the excitatory nerons

    for (int i = 0; i < I; i++) {
        int J = m->sphist[i].size();
        // we have an array of pointers
        (m->ptr_r)[i] = 0.0;  // zero the array
        for (int j = 0; j < J; j++) {
            // the weight of the spike will be the (exponentially) lower the older it is
            float expw = exp(0.08 * ((m->sphist[i][j]) - (m->t)));
            (m->ptr_r)[i] += expw;
        }
    }
    return m->ptr_r;
}
void dumpSpikeStates(Model* m) {
    m->saveDSPTS();
    m->saveX();
    m->saveSpts();
}

void loadSpikeStates(Model* m, char* fname) {
    // this func takes a c_char_p ctypes type, dereferences it into a string
    // fname comes in as a zero-terminated pointer, so c++ knows the start and
    // stop
    cout << "loading spike states" << endl;
    m->loadDSPTS(fname);
    m->loadX(fname);
    m->loadSpts(fname);
}

void set_mex(Model* m, double _mex) { (m->mex) = _mex; }

void set_hEhI(Model* m, double _hE, double _hI) {
    (m->hE) = _hE;
    (m->hI) = _hI;
}

void set_t(Model* m, double _t) { (m->t) = _t; }

void saveSpikes(Model* m, int _saveflag) { (m->saveflag) = _saveflag; }

// this func takes a pointer to the object of type Net and calls the bar()
// method of that object because this is a pointer, to access a member of the
// class, you use an arrow, not dot
void sim(Model* m, int steps) { m->sim(steps); }

double* getWeights(Model* m) {
    const int x = m->N;
    vector<vector<double>>* arrayOfPointersVec = &(m->Jo);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < x; j++) {
            (m->ptr_Jo)[j + x * i] = (double)(*arrayOfPointersVec)[i][j];
        }
    }
    return m->ptr_Jo;
}

double* getF(Model* m) {
    const int x = m->NE;
    // because we can't return a pointer of type vector<double>*,
    // (C doesn't know this type), we have to create a new pointer
    // of type double*
    vector<double>* arrayOfPointersVec = &(m->F);
    for (int i = 0; i < x; i++) {
        (m->ptr_F)[i] = (double)(*arrayOfPointersVec)[i];
    }
    return m->ptr_F;
}

double* getD(Model* m) {
    // because we can't return a pointer of type vector<double>*,
    // (C doesn't know this type), we have to create a new pointer
    // of type double*
    const int x = m->NE;
    vector<double>* arrayOfPointersVec = &(m->D);
    for (int i = 0; i < x; i++) {
        (m->ptr_D)[i] = (double)(*arrayOfPointersVec)[i];
    }
    return m->ptr_D;
}

void setParams(Model* m, Model::ParamsStructType params) { m->setParams(params); }

void setWeights(Model* m, double* W) {
    // if a function takes a pointer the * symbol means that the address
    // gets automatically dereferenced
    const int x = m->N;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < x; j++) {
            // we take a pointer, dereference the Jo, write to [i, j]
            // dereferenced values of W[i,j] (remember, square brackets
            // dereference a pointer)
            // it seems that only pointer of type <vector<double>> can be
            // addressed like 2d arrays
            (m->Jo)[i][j] = W[j + x * i];
        }
    }
    cout << "Weights set" << endl;
    m->reinit_Jinidx();
    cout << "Jinidx recalculated" << endl;
}

void setF(Model* m, double* F) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the F, write to [i, j]
        // dereferenced values of F[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->F)[i] = F[i];
    }
    // 	cout << "F loaded" << endl;
}

void setD(Model* m, double* D) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the D, write to [i]
        // dereferenced values of D[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->D)[i] = D[i];
    }
    // 	cout << "D loaded" << endl;
}

void setStim(Model* m, int* hStim) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the ys, write to [i]
        // dereferenced values of ys[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->hStim)[i] = hStim[i];
    }
    // cout << "stim set" << endl;
}

void perturbU(Model* m) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the F, write to [i, j]
        // dereferenced values of F[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->UU)[i] = (m->ngn()) * 0.1;
        if ((m->UU)[i] < 0.0) {
            (m->UU)[i] *= -1;
        }
    }
}

double* getUU(Model* m) {
    // because we can't return a pointer of type vector<double>*,
    // (C doesn't know this type), we have to create a new pointer
    // of type double*
    const int x = m->NE;
    vector<double>* arrayOfPointersVec = &(m->UU);
    for (int i = 0; i < x; i++) {
        (m->ptr_UU)[i] = (double)(*arrayOfPointersVec)[i];
    }
    return m->ptr_UU;
}

void setUU(Model* m, double* UU) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the F, write to [i, j]
        // dereferenced values of F[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->UU)[i] = UU[i];
    }
}

double* getTheta(Model* m) {
    // because we can't return a pointer of type vector<double>*,
    // (C doesn't know this type), we have to create a new pointer
    // of type double*
    const int x = m->NE;
    vector<double>* arrayOfPointersVec = &(m->theta);
    for (int i = 0; i < x; i++) {
        (m->ptr_theta)[i] = (double)(*arrayOfPointersVec)[i];
    }
    return m->ptr_theta;
}

void setStimIntensity(Model* m, double* _stimIntensity) {
    const int x = m->NE;
    for (int i = 0; i < x; i++) {
        // we take a pointer, dereference the F, write to [i, j]
        // dereferenced values of F[i] (remember, square brackets
        // dereference a pointer)
        // it seems that only pointer of type <vector<double>> can be addressed
        // like 2d arrays
        (m->stimIntensity)[i] = _stimIntensity[i];
        // cout << _stimIntensity[i] << " ";
    }
}

void set_Ip(Model* m, double _Ip) { (m->Ip) = _Ip; }

void set_symmetric(Model* m, bool _symmetric) { (m->symmetric) = _symmetric; }

bool get_symmetric(Model* m, bool _symmetric) { return m->symmetric; }

void set_STDP(Model* m, bool _STDPon) { (m->STDPon) = _STDPon; }

bool get_STDP(Model* m) { return m->STDPon; }

void set_homeostatic(Model* m, bool _homeostatic) { (m->homeostatic) = _homeostatic; }

bool get_homeostatic(Model* m) { return m->homeostatic; }

void set_HAGA(Model* m, bool _HAGA) { (m->HAGA) = _HAGA; }

double calcMeanW(Model* m, int* iIDs, int leniIDs, int* jIDs, int lenjIDs) {
    double _acc = 0;
    int _c = 0;
    // we take a pointer, dereference the ys, write to [i]
    // dereferenced values of ys[i] (remember, square brackets
    // dereference a pointer)
    // it seems that only pointer of type <vector<double>> can be addressed like
    // 2d arrays

    for (int i = 0; i < leniIDs; i++) {
        for (int j = 0; j < lenjIDs; j++) {
            if ((m->Jo)[iIDs[i]][jIDs[j]] > (m->Jepsilon)) {
                _c += 1;
                _acc += (m->Jo)[iIDs[i]][jIDs[j]];
            }
        }
    }
    return _acc / _c;
}

double calcMeanF(Model* m, int* iIDs, int leniIDs) {
    double _acc = 0;
    int _c = 0;
    // we take a pointer, dereference the ys, write to [i]
    // dereferenced values of ys[i] (remember, square brackets
    // dereference a pointer)
    // it seems that only pointer of type <vector<double>> can be addressed like
    // 2d arrays

    for (int i = 0; i < leniIDs; i++) {
        _c += 1;
        _acc += (m->F)[iIDs[i]];
    }
    return _acc / _c;
}

double calcMeanD(Model* m, int* iIDs, int leniIDs) {
    double _acc = 0;
    int _c = 0;
    // we take a pointer, dereference the ys, write to [i]
    // dereferenced values of ys[i] (remember, square brackets
    // dereference a pointer)
    // it seems that only pointer of type <vector<double>> can be addressed like
    // 2d arrays

    for (int i = 0; i < leniIDs; i++) {
        _c += 1;
        _acc += (m->D)[iIDs[i]];
    }
    return _acc / _c;
}

Model::retParamsStructType getState(Model* m) { return m->getState(); }
}