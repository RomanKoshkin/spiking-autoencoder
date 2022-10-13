// Reduced binary model of short- and long-term synaptic plasticity
//
// Original code by Created by Naoki Hiratani (N.Hiratani@gmail.com)
// Comments and Python-friendly interactive implementation
// by Roman Koshkin (roman.koshkin@gmail.com)

// добавил alphaLTP, alphaLTD к функции STDP

#include <algorithm>
#include <boost/format.hpp>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// to complile
// g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_9_haga_grid.cpp -o ./bmm.dylib
// g++ -std=gnu++11 -O3 -dynamiclib -ftree-vectorize -march=native -mavx bmm_7_haga.cpp -o ./bmm.dylib
// sudo /usr/bin/g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx bmm_8_haga.cpp -o ./bmm.dylib
// icc -std=gnu++11 -O3 -shared -fPIC bmm_5_haga.cpp -o ./bmm.dylib
// g++ -std=gnu++11 -Ofast -shared -fPIC -ftree-vectorize -march=native -mavx -fopenmp bmm_9_haga.cpp -o ./bmm.dylib

struct Timer {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	std::chrono::duration<float> duration;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	/* when the function where this object is created returns,
	this object must be destroyed, hence this destructor is called */
	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		float ms = duration.count() * 1000.0f;
		// std::cout << "Elapsed (c++ timer): " << ms << " ms." << std::endl;
	}
};

// class definition
class Model {
public:
	// input struct

	typedef struct
	{
		double alpha;
		double JEI;

		double T;
		double h;

		// int NE;
		// int NI;

		// probability of connection
		double cEE;
		double cIE;
		double cEI;
		double cII;

		// Synaptic weights
		double JEE;
		double JEEinit;
		double JIE;
		double JII;

		//initial conditions of synaptic weights
		double JEEh;
		double sigJ;

		double Jtmax;
		double Jtmin;

		// Thresholds of update
		double hE;
		double hI;

		double IEex;
		double IIex;
		double mex;
		double sigex;

		// Average intervals of update, ms
		double tmE;
		double tmI;

		//Short-Term Depression
		double trec;
		double Jepsilon;

		// Time constants of STDP decay
		double tpp;
		double tpd;
		double twnd;

		// Coefficients of STDP
		double g;

		//homeostatic
		int itauh;

		double hsd;
		double hh;

		double Ip;
		double a;

		double xEinit;
		double xIinit;
		double tinit;

		double U;
		double taustf;
		double taustd;
		double Cp;
		double Cd;
		bool HAGA;
	} ParamsStructType;

	// returned struct
	typedef struct
	{
		double alpha;
		double JEI;

		double T;
		double h;

		int NE;
		int NI;

		// probability of connection
		double cEE;
		double cIE;
		double cEI;
		double cII;

		// Synaptic weights
		double JEE;
		double JEEinit;
		double JIE;
		double JII;

		//initial conditions of synaptic weights
		double JEEh;
		double sigJ;

		double Jtmax;
		double Jtmin;

		// Thresholds of update
		double hE;
		double hI;

		double IEex;
		double IIex;
		double mex;
		double sigex;

		// Average intervals of update, ms
		double tmE;
		double tmI;

		//Short-Term Depression
		double trec;
		double Jepsilon;

		// Time constants of STDP decay
		double tpp;
		double tpd;
		double twnd;

		// Coefficients of STDP
		double g;

		//homeostatic
		int itauh;

		double hsd;
		double hh;

		double Ip;
		double a;

		double xEinit;
		double xIinit;
		double tinit;

		double Jmin;
		double Jmax;
		double Cp;
		double Cd;
		int SNE;
		int SNI;
		int NEa;
		double t;

		double U;
		double taustf;
		double taustd;
		bool HAGA;
		int cell_id;
	} retParamsStructType;

	// ostringstream ossSTP;
	// ofstream ofsSTP;
	double accum;

	int nstim;
	int NE, NI, N, NEo;
	int SNE, SNI; //how many neurons get updated per time step
	int NEa;			// Exact number of excitatory neurons stimulated externally
	int pmax;

	vector<vector<double>> Jo;
	vector<vector<double>> Ji;

	double alpha = 50.0; // Degree of log-STDP (50.0)
	double JEI = 0.15;	 // 0.15 or 0.20

	double pi = 3.14159265;
	double e = 2.71828182;

	double T = 1800 * 1000.0; // simulation time, ms
	double h = 0.01;					// time step, ms ??????

	// probability of connection
	double cEE = 0.2; //
	double cIE = 0.2; //
	double cEI = 0.5; //
	double cII = 0.5; //

	// Synaptic weights
	double JEE = 0.15;		 //
	double JEEinit = 0.15; // ?????????????
	double JIE = 0.15;		 //
	double JII = 0.06;		 //
	//initial conditions of synaptic weights
	double JEEh = 0.15; // Standard synaptic weight E-E
	double sigJ = 0.3;	//

	double Jtmax = 0.25; // J_maxˆtot
	double Jtmin = 0.01; // J_minˆtot // ??? NOT IN THE PAPER

	// WEIGHT CLIPPING     // ???
	double Jmax = 5.0 * JEE;	// ???
	double Jmin = 0.01 * JEE; // ????

	// Thresholds of update
	double hE = 1.0; // Threshold of update of excitatory neurons
	double hI = 1.0; // Threshold of update of inhibotory neurons

	double IEex = 2.0;	// Amplitude of steady external input to excitatory neurons
	double IIex = 0.5;	// Amplitude of steady external input to inhibitory neurons
	double mex = 0.3;		// mean of external input
	double sigex = 0.1; // variance of external input

	// Average intervals of update, ms
	double tmE = 5.0; //t_Eud EXCITATORY
	double tmI = 2.5; //t_Iud INHIBITORY

	//Short-Term Depression
	double trec = 600.0; // recovery time constant (tau_sd, p.13 and p.12)
	//double usyn = 0.1;
	double Jepsilon = 0.001; // BEFORE UPDATING A WEIGHT, WE CHECK IF IT IS GREATER THAN
													 // Jepsilon. If smaller, we consider this connection as
													 // non-existent, and do not update the weight.

	// Time constants of STDP decay
	double tpp = 20.0;	 // tau_p
	double tpd = 40.0;	 // tau_d
	double twnd = 500.0; // STDP window lenght, ms

	// Coefficients of STDP
	double Cp = 0.1 * JEE;			// must be 0.01875 (in the paper)
	double Cd = Cp * tpp / tpd; // must be 0.0075 (in the paper)

	//homeostatic
	//double hsig = 0.001*JEE/sqrt(10.0);
	double hsig = 0.001 * JEE; // i.e. 0.00015 per time step (10 ms)
	int itauh = 100;					 // decay time of homeostatic plasticity, (100s)

	double hsd = 0.1; // is is the timestep of integration for calculating STP
	double hh = 10.0; // SOME MYSTERIOUS PARAMETER

	double Ip = 1.0; // External current applied to randomly chosen excitatory neurons
	double a = 0.20; // Fraction of neurons to which this external current is applied

	double xEinit = 0.02; // the probability that an excitatory neurons spikes at the beginning of the simulation
	double xIinit = 0.01; // the probability that an inhibitory neurons spikes at the beginning of the simulation
	double tinit = 100.0; // period of time after which STDP kicks in

	bool STDPon = true;
	bool homeostatic = true;

	vector<double> dvec;
	vector<double> UU;

	vector<int> ivec;
	deque<double> ideque; // <<<< !!!!!!!!

	vector<deque<double>> dspts; // <<<< !!!!!!!!
	vector<int> x;
	set<int> spts;

	int saveflag = 0;
	double t = 0;
	int tidx = -1;
	// bool trtof = true; // ?????? some flag
	double u;
	int j;
	vector<int> smpld;
	set<int>::iterator it;
	double k1, k2, k3, k4;
	// bool Iptof = true;

	vector<vector<int>> Jinidx; /* list (len=3000) of lists. Each
        list lists indices of excitatory neurons whose weights are > Jepsilon */

	// classes to stream data to files
	ofstream ofsr;

	double tauh; // decay time of homeostatic plasticity, in ms
	double g;

	// method declarations
	Model(int, int, int, int); // construction
	double dice();
	double ngn();
	void sim(int);
	void setParams(ParamsStructType);
	retParamsStructType getState();
	vector<vector<double>> calc_J(double, double);
	vector<int> rnd_sample(int, int);
	double fd(double, double);

	vector<double> theta;
	vector<double> F;
	vector<double> D;
	double U = 0.6; // default release probability for HAGA
	double taustf;
	double taustd;
	bool HAGA;

	double alpha_tau = 0.12;
	double heaviside(double);
	double alpha_function_LTP(double);
	double alpha_function_LTD(double);
	double tanh_LTP(double);
	double tanh_LTD(double);

	void STPonSpike(int);
	void STPonNoSpike();
	void updateMembranePot(int);
	void checkIfStim(int);
	void STDP(int);
	void saveSTP();
	void reinitFD();
	void saveX();
	void saveDSPTS();
	void loadX(string);
	void loadDSPTS(string);
	deque<double> SplitString(string);
	deque<int> SplitString_int(string);
	void reinit_Jinidx();
	void saveSpts();
	void loadSpts(string);

	double acc0;
	double acc1;
	double acc2;
	double c0;
	double c1;
	double c2;
	double c;
	double acc;
	vector<int> hStim;
	vector<double> stimIntensity;
	int cell_id;

	// ostringstream ossa;
	// ofstream ofsa;
	// string fstra;

	// diagnostics:
	//         ostringstream ossb;
	//         ofstream ofsb;
	//         string fstrb;


	// here you just declare pointers, but you must
	// ALLOCATE space on the heap for them in the class constructor
	double *ptr_Jo;
	double *ptr_F;
	double *ptr_D;
	double *ptr_UU;
	double *ptr_theta;


	// flexible arrays can only be declared at the end of the class !!
	//         double sm[];
	// double* sm;
private:
	// init random number generator
	std::random_device m_randomdevice;
	std::mt19937 m_mt;
};

// double Model::dice(){
// 	return rand()/(RAND_MAX + 1.0);
// }

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
vector<int> Model::rnd_sample(int ktmp, int Ntmp) { // when ktmp << Ntmp
	vector<int> smpld;
	int xtmp;
	bool tof;
	while (smpld.size() < ktmp) {
		xtmp = (int)floor(Ntmp * Model::dice());
		tof = true;
		// make sure that the sampled id isn't the same as any of the previous ones
		for (int i = 0; i < smpld.size(); i++) {
			if (xtmp == smpld[i]) {
				tof = false;
			}
		}
		if (tof)
			smpld.push_back(xtmp);
	}
	return smpld;
}

double Model::fd(double x, double alpha) {
	return log(1.0 + alpha * x) / log(1.0 + alpha);
}

void Model::STPonSpike(int i) {
	F[i] += UU[i] * (1 - F[i]); // U = 0.6
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
	// so to get dy, we need to multiply the right-hand side by dt in ms (the units of taustd/taustf),
	// the time step in this simulation is 0.01 ms, so our dt (hsd in this case) would be 0.01 if
	// we updated dy every time step (that is 0.01 ms). But we update every 10th timestep, so we use
	// 0.1 instead of 0.01
	if (((int)floor(t / h)) % 10 == 0) {
		for (int i = 0; i < NE; i++) {
			F[i] += hsd * (UU[i] - F[i]) / taustf; // @@ don't forget about hsd!!!
			D[i] += hsd * (1.0 - D[i]) / taustd;
			
			// if (t < 2000000) {
			// 	F[i] += hsd * (UU[i] - F[i]) / taustf; // @@ don't forget about hsd!!!
			// 	D[i] += hsd * (1.0 - D[i]) / taustd;
			// } else {
			// 	// hsd = 1.0;								// @@ don't forget about hsd!!!
			// 	k1 = (UU[i] - F[i]) / taustf;
			// 	k2 = (UU[i] - (F[i] + 0.5 * hsd * k1)) / taustf; // @@ don't forget about hsd!!!
			// 	k3 = (UU[i] - (F[i] + 0.5 * hsd * k2)) / taustf;
			// 	k4 = (UU[i] - (F[i] + hsd * k3)) / taustf;
			// 	F[i] += hsd * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

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
	// ofsDSPTS.open("DSPTS_" + std::to_string(cell_id) + "_" + std::to_string(t));
	ofsDSPTS.open("DSPTS_" + std::to_string(cell_id));
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
	ofsX.open("X_" + std::to_string(cell_id));
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
	ifstream file("DSPTS_" + std::to_string(cell_id));

	if (!file.is_open()) {
		throw "DSPTS file not found.";
	}
	string line;
	while (getline(file, line)) {
		iideque = SplitString(line.c_str());
		iideque.pop_front();
		dspts.push_back(iideque);
		// cout << i << endl;
		// cout << dspts[i].back() << endl;
	}
	file.close();
	cout << "DSPTS loaded" << endl;
}

void Model::loadX(string tt) {
	x.clear();
	deque<double> iideque;
	// ifstream file("X_" + std::to_string(cell_id) + "_" + tt);
	ifstream file("X_" + std::to_string(cell_id));
	if (!file.is_open()) {
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
	// WE update the membrane potential of the chosen excitatory neuron (eq.4, p.12)
	// -threshold of update + steady exc input * mean/var of external stim input
	if ((i >= NE-NEo) && (i < NE)) {
		u = -hE + IEex * (mex + sigex * ngn()) * 0.6; // pentagon, p.12
	} else {
		u = -hE + IEex * (mex + sigex * ngn()); // pentagon, p.12
	}
	

	//we go over all POSTsynaptic neurons that are spiking now
	for (const auto& j : spts) {
		//if a postsynaptic spiking neuron happens to be excitatory,
		if (j < NE) {
			u += F[j] * D[j] * Jo[i][j];
			//if a postsynaptic spiking neuron happens to be inhibitory,
		} else {
			u += Jo[i][j];
		}
	}
}

void Model::checkIfStim(int i) {
	if (hStim[i] == 1) {
		if (dice() < stimIntensity[i]) {
			u += Ip;
			// std::cout << "stim @ t=" << t << " on neuron " << i << " u = " << u << std::endl;
		}
		// std::cout << "stim @ t=" << t << " on neuron " << i << std::endl;
	}
}

double Model::heaviside(double x) {
    return double(x > 0);
}

double Model::alpha_function_LTP(double wgt) {
	return exp(-wgt/alpha_tau + 1.0) * heaviside(wgt) * (wgt/alpha_tau);
}

double Model::alpha_function_LTD(double wgt) {
	return exp((wgt-1)/alpha_tau + 1.0) * heaviside(wgt + 1.0) * ((1.0 - wgt)/alpha_tau);
}

double Model::tanh_LTP(double wgt) {
	return -tanh(30*(wgt-Jmax));
}

double Model::tanh_LTD(double wgt) {
	return tanh(30*wgt);
}

void Model::STDP(int i) {

	/* First (LTD), we treat the chosen neuron as PREsynaptic and loop over all the POSTSYNAPTIC excitatory 
	neurons that THE CHOSEN NEURON synapses on. Since we're at time t (and this is the latest time),
	the spikes recorded on those "POSTsynaptic" neurons will have an earlier timing than the spike
	recorded on the currently chosen neuron (that we treat as PREsynaptic). This indicates that
	the synaptic weight between this chosen neuron (presynaptic) and all the other neurons
	(postsynaptic) will decrease.  */

	for (int ip = 0; ip < NE; ip++) {
		if (Jo[ip][i] > Jepsilon && t > tinit) {
			// dspts is a deque of spiking times on the ith POSTSYNAPTIC neurons
			for (const auto& tt : dspts[ip]) {
				if (HAGA == 1) {
					// STP-dependent (new HAGA)
					// double alphaLTP = alpha_function_LTP(Jo[ip][i]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					// double alphaLTD = alpha_function_LTD(Jo[ip][i]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					double alphaLTP = tanh_LTP(Jo[ip][i]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					double alphaLTD = tanh_LTD(Jo[ip][i]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					Jo[ip][i] += F[i] * D[i] * alphaLTP * (Cp * exp((tt - t) / tpp) - alphaLTD * fd(Jo[ip][i] / JEE, alpha) * Cd * exp((tt - t) / tpd));
				} else {
					// STP-independent (new Hiratani)
					Jo[ip][i] += (Cp * exp((tt - t) / tpp) - fd(Jo[ip][i] / JEE, alpha) * Cd * exp((tt - t) / tpd));
				}
			}
			// we force the weights to be no less than Jmin, THIS WAS NOT IN THE PAPER
			if (Jo[ip][i] > Jmax)
				Jo[ip][i] = Jmax;
			if (Jo[ip][i] < Jmin)
				Jo[ip][i] = Jmin;
		}
	}

	// (LTP)

	/* Jinidx is a list of lists (shape (2500 POSTsyn, n PREsyn)). E.g. if in row 15 we have number 10,
	it means that the weight between POSTsynaptic neuron 15 and presynaptc neuron 10 is greater than Jepsilon */

	/* we treat the currently chosen neuron as POSTtsynaptic, and we loop over all the presynaptic
	neurons that synapse on the current postsynaptic neuron. At time t (the latest time, and we don't
	yet know any spikes that will happen in the future) all the spikes on the presynaptic neurons with
	id j will have an earlier timing that the spike on the currently chosen neuron i (that we treat
	as postsynaptic for now). This indicates that the weights between the chosen neuron treated as post-
	synaptic for now and all the other neurons (treated as presynaptic for now) will be potentiated.  */

	for (const auto& j : Jinidx[i]) {
		// at each loop we get the id of the jth presynaptic neuron with J > Jepsilon
		if (t > tinit) {
			for (const auto& tt : dspts[j]) {
				// we loop over all the spike times on the jth PRESYNAPTIC neuron
				if (HAGA == 1) {
					// double alphaLTP = alpha_function_LTP(Jo[i][j]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					// double alphaLTD = alpha_function_LTD(Jo[i][j]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					double alphaLTP = tanh_LTP(Jo[i][j]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					double alphaLTD = tanh_LTD(Jo[i][j]); // <<<< ALPHA <<<<<<<<<<<<<<<<<<<<<<<<<
					// STP-dependent (по новому определению Haga)
					Jo[i][j] += F[j] * D[j] * alphaLTP * (Cp * exp(-(t - tt) / tpp) - alphaLTD * fd(Jo[i][j] / JEE, alpha) * Cd * exp(-(t - tt) / tpd));
				} else {
					// STP-independent (по новому определению Hiratani: не так как было у Хиратани в статье, а просто без F*D)
					Jo[i][j] += (Cp * exp(-(t - tt) / tpp) - fd(Jo[i][j] / JEE, alpha) * Cd * exp(-(t - tt) / tpd));
				}
			}
			// we force the weights to be no more than Jmax, THIS WAS NOT IN THE PAPER
			if (Jo[i][j] > Jmax)
				Jo[i][j] = Jmax;
			if (Jo[i][j] < Jmin)
				Jo[i][j] = Jmin;
		}
	}
}

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
		// ofsb << smpld[0] << " " << smpld[1] << " " << smpld[2] << " " << smpld[3] << " " << smpld[4] << " " << std::endl;

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

			// WE RECORD A SPIKE and PERFORM AN STDP on the chosen neuron if it spikes u > 0
			if (u > theta[i]) {
				// if the POSTsynaptic neuron chosen for update exceeds the threshold, we save its ID in the set "spts"
				spts.insert(i);

				/* dspts saves the TIME of this spike to a DEQUE, such that each row id of this deque
				corresponds to the id of the spiking neuron. The row records the times at which that
				neuron emitted at spike. */
				dspts[i].push_back(t); // SHAPE: (n_postsyn x pytsyn_sp_times)
				x[i] = 1;
				// // record a spike on an EXCITATORY neuron (because STDP is only on excitatory neurons)
				if (saveflag == 1){
					ofsr << t << " " << i << endl; // record a line to file
				}

				if (STDPon) {
					STDP(i);	
				}

				if (i > NE-NEo) {
					theta[i] += 0.03;
				}
				
			}
		}

		// exponentially decaying threshold for excitatory neurons
		for (int i_ = NE-NEo; i_ < NE; i_++) {
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
					// removing a spike time from the SET of spikes on inhibitory neurons is the same as
					// subtracting them
					spts.erase(it++);
				}
				x[i] = 0;
			}

			//update the membrane potential on a chosen inhibitory neuron
			u = -hI + IIex * (mex + sigex * ngn()); // hexagon, eq.5, p.12

			for (const int& j: spts) {
				u += Jo[i][j];
			}

			// if the membrane potential on the currently chosen INHIBITORy neuron
			// is greater than the threshold, we record a spike on this neuron.
			if (u > 0) {
				spts.insert(i);
				x[i] = 1;
				// // record a spike on an INHIBITORY NEURON
				if (saveflag == 1) {
					ofsr << t << " " << i << endl;
				}
			}
		}

		STPonNoSpike();

		// EVERY 10 ms Homeostatic plasticity, weight clipping, boundary conditions, old spike removal
		if (((int)floor(t / h)) % 1000 == 0) {

			//Homeostatic Depression
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
							Jo[i][j] = Jmin; // ????? Jmin is zero, not 0.0015, as per Table 1
						if (Jo[i][j] > Jmax)
							Jo[i][j] = Jmax;
					}
				}

				//boundary condition
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
							
							// if the total weight exceeds Jtmax, we subtract the excess value
							Jo[i][j] -= (Jav - Jtmax);
							// but if a weight is less that Jmin, we set it to Jmin (clip from below)
							if (Jo[i][j] < Jmin) {
								Jo[i][j] = Jmin;
							}
						}

					// if the total weight is less that Jtmin
					} else if (Jav < Jtmin) {
						for (const int& j : Jinidx[i]) {
							/* ???????? we top up each (!!!???) weight by the difference 
							between the total min and current total weight */
							Jo[i][j] += (Jtmin - Jav);
							// but if a weight is more that Jmax, we clip it to Jmax
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
					//if we have spike times that are occured more than 500 ms ago, we pop them from the deque
					if (t - dspts[i][0] > twnd) {
						dspts[i].pop_front();
					}
				}
			}
		}

		// EVERY 1s
		if (((int)floor(t / h)) % (1000 * 100) == 0) {

			tidx += 1; // we count the number of 1s cycles
			int s = 0;
			it = spts.begin();
			while (it != spts.end()) {
				++s;
				++it;
			}

			// exit if either no neurons are spiking or too many spiking after t > 200 ms
			if (s == 0 || (s > 1.0 * NE && t > 200.0)) {
				std::cout << "Exiting because either 0 or too many spikes at t =" << t << std::endl;
				break;
			}
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
				if (J[i][j] < Jmin)
					J[i][j] = Jmin;
				if (J[i][j] > Jmax)
					J[i][j] = Jmax;
			}
		}
		// then the E-I weights
		for (int j = NE; j < N; j++) {
			J[i].push_back(0.0); // here the matrix J is at first of size 2500, we extend it
			if (dice() < cEI) {
				J[i][j] -= JEI; /* becuase jth presynaptic inhibitory synapsing on
				an ith excitatory postsynaptic neuron should inhibit it. Hence the minus */
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
	for (int i = NE-NEo; i < N; i++){ // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  NE-NEo or N
		for (int j = NE-NEo; j < NE; j++){
			J[i][j] = 0.0;
		}
	}
	for (int i = NE-NEo; i < NE; i++){
		for (int j = NE-NEo; j < N; j++){
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
		Jinidx.push_back(ivec); // shape = (3000, max3000)
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
	ofsSpts.open("SPTS_" + to_string(cell_id));

	for (int spt : spts) {
		ofsSpts << spt << " ";
	}
}

void Model::loadSpts(string tt) {
	spts.clear();

	deque<int> iideque;
	// ifstream file("SPTS_" + to_string(cell_id) + "_" + tt);
	ifstream file("SPTS_" + to_string(cell_id));

	if (!file.is_open()) {
		throw "X file not found.";
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

// class construction _definition_. Requires no type specifiction.
Model::Model(int _NE, int _NI, int _NEo, int _cell_id) : m_mt(m_randomdevice()) {

	cell_id = _cell_id;
	NE = _NE;
	NEo = _NEo;
	NI = _NI;
	N = NE + NI; //

	// initialize the weight matrix Jo
	Jo = calc_J(JEEinit, JEI);

	for (int i = 0; i < NE; i++) {
		stimIntensity.push_back(0.0);
	}

	// initialize the STF and STD vectors
	for (int i = 0; i < NE; i++) {
		F.push_back(U);
		D.push_back(1.0);
		UU.push_back(U);
		theta.push_back(0.0); // firing thresholds for excitatory neurons;
	}

	// allocate heap memory for the array of pointers (once per model instantiation)
	ptr_Jo = new double[N * N]; // ?
	ptr_F = new double[NE];
	ptr_D = new double[NE];
	ptr_UU = new double[NE];
	ptr_theta = new double[NE];

	// since no suitable conversion function from "std::vector<double, std::allocator<double>>"
	// to "double *" exists, we need to copy the addresses one by one
	for (int i = 0; i < NE; i++) {
		ptr_F[i] = F[i];
		ptr_D[i] = D[i];
		ptr_UU[i] = UU[i];
		ptr_theta[i] = theta[i];
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			ptr_Jo[j + N * i] = Jo[i][j];
		}
	}

	//how many neurons get updated per time step
	SNE = (int)floor(NE * h / tmE + 0.001);
	SNI = max(1, (int)floor(NI * h / tmI + 0.001));

	NEa = (int)floor(NE * a + 0.01); // Exact number of excitatory neurons stimulated externally
	pmax = NE / NEa;

	srand((unsigned int)time(NULL));

	// ossSpike << "stp" << ".txt";

	// spts_fname = "spike_times_" + cell_id;
	ofsr.open("spike_times_" + std::to_string(cell_id));
	ofsr.precision(10);

	for (int i = 0; i < NE; i++) {
		Jinidx.push_back(ivec); // shape = (3000, max3000)
		for (int i2 = 0; i2 < NE; i2++) {
			if (Jo[i][i2] > Jepsilon) {
				Jinidx[i].push_back(i2);
			}
		}
	}

	// create a vector size N and fill it with zeros
	// this vector says if a neuron is spiking or not
	for (int i = 0; i < N; i++) {
		x.push_back(0);
	}

	// we remember in spts the ids of neurons that are spiking at the current step
	// and set neurons with these ids to 1
	// at the beginning of the simulation, some neurons have to spike, so we
	// initialize some neurons to 1 to make them spike
	for (int i = 0; i < N; i++) {
		// elements corresponding to excitatory neurons are filled with
		// ones with probability xEinit (0.02)
		if (i < NE && dice() < xEinit) {
			spts.insert(i);
			x[i] = 1;
		}
		// elements corresponding to inhibitory neurons are filled with
		// ones with probability xIinit (0.01)
		if (i >= NE && dice() < xIinit) {
			spts.insert(i);
			x[i] = 1;
		}
	}

	for (int i = 0; i < NE; i++) {
		dspts.push_back(ideque);
	}

	// initialize stimulus
	for (int i = 0; i < NE; i++) {
		hStim.push_back(0);
	}
}

void Model::setParams(Model::ParamsStructType params) {
	alpha = params.alpha;				// Degree of log-STDP (50.0)
	JEI = params.JEI;						// 0.15 or 0.20
	T = params.T;								// simulation time, ms (1800*1000.0)
	h = params.h;								// time step, ms ??????
	cEE = params.cEE;						// 0.2
	cIE = params.cIE;						// 0.2
	cEI = params.cEI;						// 0.5
	cII = params.cII;						// 0.5
	JEE = params.JEE;						// 0.15
	JEEinit = params.JEEinit;		// 0.18
	JIE = params.JIE;						// 0.15
	JII = params.JII;						// 0.06
	JEEh = params.JEEh;					// Standard synaptic weight E-E 0.15
	sigJ = params.sigJ;					// 0.3
	Jtmax = params.Jtmax;				// J_maxˆtot (0.25)
	Jtmin = params.Jtmin;				// J_minˆtot // ??? NOT IN THE PAPER (0.01)
	hE = params.hE;							// Threshold of update of excitatory neurons 1.0
	hI = params.hI;							// Threshold of update of inhibotory neurons 1.0
	IEex = params.IEex;					// Amplitude of steady external input to excitatory neurons 2.0
	IIex = params.IIex;					// Amplitude of steady external input to inhibitory neurons 0.5
	mex = params.mex;						// mean of external input 0.3
	sigex = params.sigex;				// variance of external input 0.1
	tmE = params.tmE;						//t_Eud EXCITATORY 5.0
	tmI = params.tmI;						//t_Iud INHIBITORY 2.5
	trec = params.trec;					// recovery time constant (tau_sd, p.13 and p.12) 600.0
	Jepsilon = params.Jepsilon; // ???????? 0.001
	tpp = params.tpp;						// tau_p 20.0
	tpd = params.tpd;						// tau_d 40.0
	Cp = params.Cp;
	Cd = params.Cd;
	twnd = params.twnd;			// STDP window lenght, ms 500.0
	g = params.g;						// ???? 1.25
	itauh = params.itauh;		// decay time of homeostatic plasticity,s (100)
	hsd = params.hsd;				// integration time step (10 * simulation timestep)
	hh = params.hh;					// SOME MYSTERIOUS PARAMETER 10.0
	Ip = params.Ip;					// External current applied to randomly chosen excitatory neurons 1.0
	a = params.a;						// Fraction of neurons to which this external current is applied 0.20
	xEinit = params.xEinit; // prob that an exc neurons spikes at the beginning of the simulation 0.02
	xIinit = params.xIinit; // prob that an inh neurons spikes at the beginning of the simulation 0.01
	tinit = params.tinit;		// period of time after which STDP kicks in 100.0

	// recalculate values that depend on the parameters
	SNE = (int)floor(NE * h / tmE + 0.001);
	SNI = max(1, (int)floor(NI * h / tmI + 0.001));

	Jmax = 5.0 * JEE;	 // ???
	Jmin = 0.01 * JEE; // ????
	// Cp = 0.1*JEE;              // must be 0.01875 (in the paper)
	// Cd = Cp*tpp/tpd;           // must be 0.0075 (in the paper)
	hsig = 0.001 * JEE;							 // i.e. 0.00015 per time step (10 ms)
	NEa = (int)floor(NE * a + 0.01); // Exact number of excitatory neurons stimulated externally
	pmax = NE / NEa;

	tauh = itauh * 1000.0; // decay time of homeostatic plasticity, in ms

	U = params.U;

	taustf = params.taustf;
	taustd = params.taustd;
	HAGA = params.HAGA;

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

	return ret_struct;
}

extern "C" {

// we create a pointer to the object of type Net. The pointer type must be of the same type as the
// object/variable this pointer points to
Model *createModel(int _NE, int _NI, int _NEo, int _cell_id) {
	cout << _NE << " " << _NI << " " << _NEo << " " << _cell_id << " " << endl;
	return new Model(_NE, _NI, _NEo, _cell_id);
}

void dumpSpikeStates(Model *m) {
	m->saveDSPTS();
	m->saveX();
	m->saveSpts();
}

void loadSpikeStates(Model *m, char *fname) {
	// this func takes a c_char_p ctypes type, dereferences it into a string
	// fname comes in as a zero-terminated pointer, so c++ knows the start and stop
	cout << "loading spike states" << endl;
	m->loadDSPTS(fname);
	m->loadX(fname);
	m->loadSpts(fname);
}

void set_mex(Model *m, double _mex) {
	(m->mex) = _mex;
}

void set_hEhI(Model *m, double _hE, double _hI) {
	(m->hE) = _hE;
	(m->hI) = _hI;
}

void set_t(Model *m, double _t) {
	(m->t) = _t;
}

void saveSpikes(Model *m, int _saveflag) {
	(m->saveflag) = _saveflag;
}

// this func takes a pointer to the object of type Net and calls the bar() method of that object
// because this is a pointer, to access a member of the class, you use an arrow, not dot
void sim(Model *m, int steps) {
	m->sim(steps);
}

double *getWeights(Model *m) {
	const int x = m->N;
	vector<vector<double>> *arrayOfPointersVec = &(m->Jo);
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < x; j++) {
			(m->ptr_Jo)[j + x * i] = (double)(*arrayOfPointersVec)[i][j];
		}
	}
	return m->ptr_Jo;
}

double *getF(Model *m) {
	const int x = m->NE;
	// because we can't return a pointer of type vector<double>*,
	// (C doesn't know this type), we have to create a new pointer
	// of type double*
	vector<double> *arrayOfPointersVec = &(m->F);
	for (int i = 0; i < x; i++) {
		(m->ptr_F)[i] = (double)(*arrayOfPointersVec)[i];
	}
	return m->ptr_F;
}

double *getD(Model *m) {
	// because we can't return a pointer of type vector<double>*,
	// (C doesn't know this type), we have to create a new pointer
	// of type double*
	const int x = m->NE;
	vector<double> *arrayOfPointersVec = &(m->D);
	for (int i = 0; i < x; i++) {
		(m->ptr_D)[i] = (double)(*arrayOfPointersVec)[i];
	}
	return m->ptr_D;
}

void setParams(Model *m, Model::ParamsStructType params) {
	m->setParams(params);
}

void setWeights(Model *m, double *W) {
	// if a function takes a pointer the * symbol means that the address
	// gets automatically dereferenced
	const int x = m->N;
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < x; j++) {
			// we take a pointer, dereference the Jo, write to [i, j]
			// dereferenced values of W[i,j] (remember, square brackets
			// dereference a pointer)
			// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
			(m->Jo)[i][j] = W[j + x * i];
		}
	}
	cout << "Weights set" << endl;
	m->reinit_Jinidx();
	cout << "Jinidx recalculated" << endl;
}

void setF(Model *m, double *F) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the F, write to [i, j]
		// dereferenced values of F[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->F)[i] = F[i];
	}
// 	cout << "F loaded" << endl;

}

void setD(Model *m, double *D) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the D, write to [i]
		// dereferenced values of D[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->D)[i] = D[i];
	}
// 	cout << "D loaded" << endl;

}

void setStim(Model *m, int *hStim) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the ys, write to [i]
		// dereferenced values of ys[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->hStim)[i] = hStim[i];
	}
	// cout << "stim set" << endl;
}

void perturbU(Model *m) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the F, write to [i, j]
		// dereferenced values of F[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->UU)[i] = (m->ngn()) * 0.1;
		if ((m->UU)[i] < 0.0) {
			(m->UU)[i] *= -1;
		} 
	}
}

double *getUU(Model *m) {
	// because we can't return a pointer of type vector<double>*,
	// (C doesn't know this type), we have to create a new pointer
	// of type double*
	const int x = m->NE;
	vector<double> *arrayOfPointersVec = &(m->UU);
	for (int i = 0; i < x; i++) {
		(m->ptr_UU)[i] = (double)(*arrayOfPointersVec)[i];
	}
	return m->ptr_UU;
}

void setUU(Model *m, double *UU) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the F, write to [i, j]
		// dereferenced values of F[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->UU)[i] = UU[i];
	}
}

double *getTheta(Model *m) {
	// because we can't return a pointer of type vector<double>*,
	// (C doesn't know this type), we have to create a new pointer
	// of type double*
	const int x = m->NE;
	vector<double> *arrayOfPointersVec = &(m->theta);
	for (int i = 0; i < x; i++) {
		(m->ptr_theta)[i] = (double)(*arrayOfPointersVec)[i];
	}
	return m->ptr_theta;
}

void setStimIntensity(Model *m, double *_stimIntensity) {
	const int x = m->NE;
	for (int i = 0; i < x; i++) {
		// we take a pointer, dereference the F, write to [i, j]
		// dereferenced values of F[i] (remember, square brackets
		// dereference a pointer)
		// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
		(m->stimIntensity)[i] = _stimIntensity[i];
		// cout << _stimIntensity[i] << " ";
	}
}

void set_Ip(Model *m, double _Ip) {
	(m->Ip) = _Ip;
}

void set_STDP(Model *m, bool _STDPon) {
	(m->STDPon) = _STDPon;
}

bool get_STDP(Model *m) {
	return m->STDPon;
}

void set_homeostatic(Model *m, bool _homeostatic) {
	(m->homeostatic) = _homeostatic;
}

bool get_homeostatic(Model *m) {
	return m->homeostatic;
}

void set_HAGA(Model *m, bool _HAGA) {
	(m->HAGA) = _HAGA;
}

double calcMeanW(Model *m, int *iIDs, int leniIDs, int *jIDs, int lenjIDs) {
	
	double _acc = 0;
	int _c = 0;
	// we take a pointer, dereference the ys, write to [i]
	// dereferenced values of ys[i] (remember, square brackets
	// dereference a pointer)
	// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
	
	for (int i = 0; i < leniIDs; i++) {
		for (int j = 0; j < lenjIDs; j++) {
			if ((m->Jo)[iIDs[i]][jIDs[j]] > (m->Jepsilon)){
				_c += 1;
				_acc += (m->Jo)[iIDs[i]][jIDs[j]];
			}
		}
	}
	return _acc/_c;
}

double calcMeanF(Model *m, int *iIDs, int leniIDs) {
	
	double _acc = 0;
	int _c = 0;
	// we take a pointer, dereference the ys, write to [i]
	// dereferenced values of ys[i] (remember, square brackets
	// dereference a pointer)
	// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
	
	for (int i = 0; i < leniIDs; i++) {
		_c += 1;
		_acc += (m->F)[iIDs[i]];
	}
	return _acc/_c;
}

double calcMeanD(Model *m, int *iIDs, int leniIDs) {
	
	double _acc = 0;
	int _c = 0;
	// we take a pointer, dereference the ys, write to [i]
	// dereferenced values of ys[i] (remember, square brackets
	// dereference a pointer)
	// it seems that only pointer of type <vector<double>> can be addressed like 2d arrays
	
	for (int i = 0; i < leniIDs; i++) {
		_c += 1;
		_acc += (m->D)[iIDs[i]];
	}
	return _acc/_c;
}

Model::retParamsStructType getState(Model *m) {
	return m->getState();
}
}
