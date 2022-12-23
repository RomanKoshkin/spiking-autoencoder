#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::experimental::filesystem;

string getexepath() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

// class definition
class Model {
   public:
    // input struct

    typedef struct {
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

        // initial conditions of synaptic weights
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

        // Short-Term Depression
        double trec;
        double Jepsilon;

        // Time constants of STDP decay
        double tpp;
        double tpd;
        double twnd;

        // Coefficients of STDP
        double g;

        // homeostatic
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
        bool symmetric;
    } ParamsStructType;

    // returned struct
    typedef struct {
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

        // initial conditions of synaptic weights
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

        // Short-Term Depression
        double trec;
        double Jepsilon;

        // Time constants of STDP decay
        double tpp;
        double tpd;
        double twnd;

        // Coefficients of STDP
        double g;

        // homeostatic
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
        // int cell_id;
        bool symmetric;
    } retParamsStructType;

    // ostringstream ossSTP;
    // ofstream ofsSTP;
    double accum;

    int nstim;
    int NE, NI, N, NEo;
    int SNE, SNI;  // how many neurons get updated per time step
    int NEa;       // Exact number of excitatory neurons stimulated externally
    int pmax;

    vector<vector<double>> Jo;
    vector<vector<double>> Ji;

    double alpha = 50.0;  // Degree of log-STDP (50.0)
    double JEI = 0.15;    // 0.15 or 0.20

    double pi = 3.14159265;
    double e = 2.71828182;

    double T = 1800 * 1000.0;  // simulation time, ms
    double h = 0.01;           // time step, ms ??????

    // probability of connection
    double cEE = 0.2;  //
    double cIE = 0.2;  //
    double cEI = 0.5;  //
    double cII = 0.5;  //

    // Synaptic weights
    double JEE = 0.15;      //
    double JEEinit = 0.15;  // ?????????????
    double JIE = 0.15;      //
    double JII = 0.06;      //
    // initial conditions of synaptic weights
    double JEEh = 0.15;  // Standard synaptic weight E-E
    double sigJ = 0.3;   //

    double Jtmax = 0.25;  // J_maxˆtot
    double Jtmin = 0.01;  // J_minˆtot // ??? NOT IN THE PAPER

    // WEIGHT CLIPPING     // ???
    double Jmax = 5.0 * JEE;   // ???
    double Jmin = 0.01 * JEE;  // ????

    // Thresholds of update
    double hE = 1.0;  // Threshold of update of excitatory neurons
    double hI = 1.0;  // Threshold of update of inhibotory neurons

    double IEex = 2.0;   // Amplitude of steady external input to excitatory neurons
    double IIex = 0.5;   // Amplitude of steady external input to inhibitory neurons
    double mex = 0.3;    // mean of external input
    double sigex = 0.1;  // variance of external input

    // Average intervals of update, ms
    double tmE = 5.0;  // t_Eud EXCITATORY
    double tmI = 2.5;  // t_Iud INHIBITORY

    // Short-Term Depression
    double trec = 600.0;  // recovery time constant (tau_sd, p.13 and p.12)
    // double usyn = 0.1;
    double Jepsilon = 0.001;  // BEFORE UPDATING A WEIGHT, WE CHECK IF IT IS GREATER THAN
                              // Jepsilon. If smaller, we consider this connection as
                              // non-existent, and do not update the weight.

    // Time constants of STDP decay
    double tpp = 20.0;    // tau_p
    double tpd = 40.0;    // tau_d
    double twnd = 500.0;  // STDP window lenght, ms

    // Coefficients of STDP
    double Cp = 0.1 * JEE;       // must be 0.01875 (in the paper)
    double Cd = Cp * tpp / tpd;  // must be 0.0075 (in the paper)

    // homeostatic
    // double hsig = 0.001*JEE/sqrt(10.0);
    double hsig = 0.001 * JEE;  // i.e. 0.00015 per time step (10 ms)
    int itauh = 100;            // decay time of homeostatic plasticity, (100s)

    double hsd = 0.1;  // is is the timestep of integration for calculating STP
    double hh = 10.0;  // SOME MYSTERIOUS PARAMETER

    double Ip = 1.0;  // External current applied to randomly chosen excitatory neurons
    double a = 0.20;  // Fraction of neurons to which this external current is applied

    double xEinit = 0.02;  // the probability that an excitatory neurons spikes
                           // at the beginning of the simulation
    double xIinit = 0.01;  // the probability that an inhibitory neurons spikes
                           // at the beginning of the simulation
    double tinit = 100.0;  // period of time after which STDP kicks in

    double DEQUE_LEN_MS = 50.0;

    bool STDPon = true;
    bool symmetric = true;
    bool homeostatic = true;

    vector<double> dvec;
    vector<double> UU;

    deque<double> spdeque;         // holds a recent history of spikes of one neurons
    vector<deque<double>> sphist;  // holds neuron-specific spike histories

    vector<int> ivec;
    deque<double> ideque;  // <<<< !!!!!!!!

    vector<deque<double>> dspts;  // <<<< !!!!!!!!
    vector<int> x;
    set<int> spts;

    // a snapshot of membrane potential values
    vector<deque<double>> Uinh;
    vector<deque<double>> Uexc;
    deque<double> fdeque;

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

    vector<vector<int>> Jinidx; /* list (len=3000) of lists. Eachlist lists indices of excitatory
                                   neurons whose weights are > Jepsilon */

    // classes to stream data to files
    ofstream ofsr;

    double tauh;  // decay time of homeostatic plasticity, in ms
    double g;

    // method declarations
    Model(int, int, int, int);  // construction
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
    double U = 0.6;  // default release probability for HAGA
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
    void symSTDP(int);
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
    void saveRecentSpikes(int, double);

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
    double* ptr_Jo;
    double* ptr_F;
    double* ptr_D;
    double* ptr_UU;
    double* ptr_theta;
    double* ptr_r;
    double* ptr_Uexc;
    double* ptr_Uinh;

    // flexible arrays can only be declared at the end of the class !!
    //         double sm[];
    // double* sm;

    string cwd;

   private:
    // init random number generator
    std::random_device m_randomdevice;
    std::mt19937 m_mt;
};

// class construction _definition_. Requires no type specifiction.
Model::Model(int _NE, int _NI, int _NEo, int _cell_id) : m_mt(m_randomdevice()) {
    cwd = getexepath();
    cout << cwd << endl;

    cell_id = _cell_id;
    NE = _NE;
    NEo = _NEo;
    NI = _NI;
    N = NE + NI;  //

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
        theta.push_back(0.0);  // firing thresholds for excitatory neurons;
    }

    // allocate heap memory for the array of pointers (once per model
    // instantiation)
    ptr_Jo = new double[N * N];  // ?
    ptr_F = new double[NE];
    ptr_D = new double[NE];
    ptr_UU = new double[NE];
    ptr_theta = new double[NE];
    ptr_r = new double[NE];
    ptr_Uexc = new double[NE + NI];
    ptr_Uinh = new double[NE + NI];

    // since no suitable conversion function from "std::vector<double,
    // std::allocator<double>>" to "double *" exists, we need to copy the
    // addresses one by one
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

    // how many neurons get updated per time step
    SNE = (int)floor(NE * h / tmE + 0.001);
    SNI = max(1, (int)floor(NI * h / tmI + 0.001));

    NEa = (int)floor(NE * a + 0.01);  // Exact number of excitatory neurons stimulated externally
    pmax = NE / NEa;

    srand((unsigned int)time(NULL));

    // ossSpike << "stp" << ".txt";

    // spts_fname = "spike_times_" + cell_id;
    ofsr.open("data/spike_times_" + std::to_string(cell_id));
    ofsr.precision(10);

    for (int i = 0; i < NE; i++) {
        Jinidx.push_back(ivec);  // shape = (3000, max3000)
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

    // we remember in spts the ids of neurons that are spiking at the current
    // step and set neurons with these ids to 1 at the beginning of the
    // simulation, some neurons have to spike, so we initialize some neurons to
    // 1 to make them spike
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

    // make a container for neurons' spike histories
    for (int i = 0; i < NE; i++) {
        sphist.push_back(spdeque);
    }

    for (int i = 0; i < NE + NI; i++) {
        Uexc.push_back(fdeque);
        Uinh.push_back(fdeque);
    }
}