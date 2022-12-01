import os
from ctypes import *
from numpy.ctypeslib import ndpointer
import numpy as np
from termcolor import cprint

cprint(os.getcwd(), color='yellow')
lib = cdll.LoadLibrary('modules/bmm.dylib')
# lib = cdll.LoadLibrary('modules/bmm.dylib')

# here we CREATE A CUSTOM C TYPE
# StimMat = c_double * 12 # here you must specify the size
# with a POINTER, you can pass arrays to C, without specifying the size of the array
StimMat = POINTER(c_double)  # https://stackoverflow.com/a/23248168/5623100


class Params(Structure):
    _fields_ = [("alpha", c_double), ("JEI", c_double), ("T", c_double), ("h", c_double), ("cEE", c_double),
                ("cIE", c_double), ("cEI", c_double), ("cII", c_double), ("JEE", c_double), ("JEEinit", c_double),
                ("JIE", c_double), ("JII", c_double), ("JEEh", c_double), ("sigJ", c_double), ("Jtmax", c_double),
                ("Jtmin", c_double), ("hE", c_double), ("hI", c_double), ("IEex", c_double), ("IIex", c_double),
                ("mex", c_double), ("sigex", c_double), ("tmE", c_double), ("tmI", c_double), ("trec", c_double),
                ("Jepsilon", c_double), ("tpp", c_double), ("tpd", c_double), ("twnd", c_double), ("g", c_double),
                ("itauh", c_int), ("hsd", c_double), ("hh", c_double), ("Ip", c_double), ("a", c_double),
                ("xEinit", c_double), ("xIinit", c_double), ("tinit", c_double), ("U", c_double), ("taustf", c_double),
                ("taustd", c_double), ("Cp", c_double), ("Cd", c_double), ("HAGA", c_bool),
                ("symmetric", c_bool)]  # https://stackoverflow.com/a/23248168/5623100]


class retParams(Structure):
    _fields_ = [("alpha", c_double), ("JEI", c_double), ("T", c_double), ("h", c_double), ("NE", c_int), ("NI", c_int),
                ("cEE", c_double), ("cIE", c_double), ("cEI", c_double), ("cII", c_double), ("JEE", c_double),
                ("JEEinit", c_double), ("JIE", c_double), ("JII", c_double), ("JEEh", c_double), ("sigJ", c_double),
                ("Jtmax", c_double), ("Jtmin", c_double), ("hE", c_double), ("hI", c_double), ("IEex", c_double),
                ("IIex", c_double), ("mex", c_double), ("sigex", c_double), ("tmE", c_double), ("tmI", c_double),
                ("trec", c_double), ("Jepsilon", c_double), ("tpp", c_double), ("tpd", c_double), ("twnd", c_double),
                ("g", c_double), ("itauh", c_int), ("hsd", c_double), ("hh", c_double), ("Ip", c_double),
                ("a", c_double), ("xEinit", c_double), ("xIinit", c_double), ("tinit", c_double), ("Jmin", c_double),
                ("Jmax", c_double), ("Cp", c_double), ("Cd", c_double), ("SNE", c_int), ("SNI", c_int), ("NEa", c_int),
                ("t", c_double), ("U", c_double), ("taustf", c_double), ("taustd", c_double), ("HAGA", c_bool),
                ("symmetric", c_bool)]


class cClassOne(object):

    # we have to specify the types of arguments and outputs of each function in the c++ class imported
    # the C types must match.

    def __init__(self, NE=-1, NI=-1, NEo=0, cell_id=-1):

        N = NE + NI
        self.NE = NE
        self.NEo = NEo
        self.params_c_obj = Params()
        self.ret_params_c_obj = retParams()

        lib.createModel.argtypes = [c_int, c_int, c_int, c_int]  # if the function gets no arguments, use None
        lib.createModel.restype = c_void_p  # returns a pointer of type void

        lib.sim.argtypes = [c_void_p, c_int]  # takes no args
        lib.sim.restype = c_void_p  # returns a void pointer

        lib.setParams.argtypes = [c_void_p, Structure]  # takes no args
        lib.setParams.restype = c_void_p  # returns a void pointer

        lib.getState.argtypes = [c_void_p]  # takes no args
        lib.getState.restype = retParams  # returns a void pointer

        lib.getWeights.argtypes = [c_void_p]  # takes no args
        lib.getWeights.restype = ndpointer(dtype=c_double, ndim=2, shape=(N, N))

        lib.setWeights.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(N, N))]
        lib.setWeights.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.getF.argtypes = [c_void_p]  # takes no args
        lib.getF.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        lib.getD.argtypes = [c_void_p]  # takes no args
        lib.getD.restype = ndpointer(dtype=c_double, ndim=1, shape=(NE,))

        lib.setF.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setF.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.setD.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setD.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.setStim.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,))]
        lib.setStim.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.getState.argtypes = [c_void_p]  # takes no args
        lib.getState.restype = retParams  # returns a void pointer

        lib.dumpSpikeStates.argtypes = [c_void_p]  # takes no args
        lib.dumpSpikeStates.restype = c_void_p  # returns a void pointer

        lib.loadSpikeStates.argtypes = [c_void_p,
                                        c_char_p]  # c_char_p is a zero-terminated pointer to a string of characters
        lib.loadSpikeStates.restype = c_void_p  # returns a void pointer

        lib.set_t.argtypes = [c_void_p, c_double]  # takes no args
        lib.set_t.restype = c_void_p  # returns a void pointer

        lib.set_Ip.argtypes = [c_void_p, c_double]  # takes no args
        lib.set_Ip.restype = c_void_p  # returns a void pointer

        lib.set_STDP.argtypes = [c_void_p, c_bool]  # takes no args
        lib.set_STDP.restype = c_void_p  # returns a void pointer

        lib.get_STDP.argtypes = [c_void_p]  # takes no args
        lib.get_STDP.restype = c_bool  # returns a void pointer

        lib.set_symmetric.argtypes = [c_void_p, c_bool]  # takes no args
        lib.set_symmetric.restype = c_void_p  # returns a void pointer

        lib.get_symmetric.argtypes = [c_void_p]  # takes no args
        lib.get_symmetric.restype = c_bool  # returns a void pointer

        lib.set_homeostatic.argtypes = [c_void_p, c_bool]  # takes no args
        lib.set_homeostatic.restype = c_void_p  # returns a void pointer

        lib.get_homeostatic.argtypes = [c_void_p]  # takes no args
        lib.get_homeostatic.restype = c_bool  # returns a void pointer

        lib.set_HAGA.argtypes = [c_void_p, c_bool]  # takes no args
        lib.set_HAGA.restype = c_void_p  # returns a void pointer

        lib.saveSpikes.argtypes = [c_void_p, c_int]  # takes no args
        lib.saveSpikes.restype = c_void_p  # returns a void pointer

        lib.perturbU.argtypes = [c_void_p]  # takes no args
        lib.perturbU.restype = c_void_p  # returns a void pointer

        lib.getUU.argtypes = [c_void_p]  # takes no args
        lib.getUU.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        lib.setUU.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setUU.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.getTheta.argtypes = [c_void_p]  # takes no args
        lib.getTheta.restype = ndpointer(dtype=np.float64, ndim=1, shape=(NE,))

        lib.setStimIntensity.argtypes = [c_void_p, ndpointer(dtype=c_double, shape=(NE,))]
        lib.setStimIntensity.restype = c_void_p  # returns a void pointer (i.e. because it doesn't have a return statement)

        lib.calcMeanW.argtypes = [
            c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int,
            ndpointer(dtype=c_int, shape=(NE,)), c_int
        ]
        lib.calcMeanW.restype = c_double  # returns a double

        lib.calcMeanF.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int]
        lib.calcMeanF.restype = c_double  # returns a double

        lib.calcMeanD.argtypes = [c_void_p, ndpointer(dtype=c_int, shape=(NE,)), c_int]
        lib.calcMeanD.restype = c_double  # returns a double

        lib.set_mex.argtypes = [c_void_p, c_double]  # takes no args
        lib.set_mex.restype = c_void_p  # returns a void pointer

        lib.set_hEhI.argtypes = [c_void_p, c_double, c_double]  # takes no args
        lib.set_hEhI.restype = c_void_p  # returns a void pointer

        # NOTE: in progress
        lib.getRecents.argtypes = [c_void_p]  # takes no args
        lib.getRecents.restype = ndpointer(dtype=c_double, shape=(NE,))

        # we call the constructor from the imported libpkg.so module
        self.obj = lib.createModel(NE, NI, NEo, cell_id)  # look in teh cpp code. CreateNet returns a pointer

    def setWeights(self, W):
        lib.setWeights(self.obj, W)

    # in the Python wrapper, you can name these methods anything you want. Just make sure
    # you call the right C methods (that in turn call the right C++ methods)

    # the order of keys defined in cluster.py IS IMPORTANT for the cClasses not to break down
    def setParams(self, params):
        for key, typ in dict(self.params_c_obj._fields_).items():
            typename = typ.__name__
            # if the current field must be c_int
            if typename == 'c_int':
                setattr(self.params_c_obj, key, c_int(params[key]))
            # if the current field must be c_double
            if typename == 'c_double':
                setattr(self.params_c_obj, key, c_double(params[key]))
            # if the current field must be c_bool
            if typename == 'c_bool':
                setattr(self.params_c_obj, key, c_bool(params[key]))
        lib.setParams(self.obj, self.params_c_obj)

    def loadSpikeStates(self, string):
        bstring = bytes(string, 'utf-8')  # you must convert a python string to bytes
        lib.loadSpikeStates(self.obj, c_char_p(bstring))

    def getState(self):
        resp = lib.getState(self.obj)
        return resp

    def getWeights(self):
        resp = lib.getWeights(self.obj)
        return resp

    def getF(self):
        resp = lib.getF(self.obj)
        return resp

    def setF(self, F):
        lib.setF(self.obj, F)

    def getD(self):
        resp = lib.getD(self.obj)
        return resp

    def setD(self, D):
        lib.setD(self.obj, D)

    def setStim(self, stimVec):
        lib.setStim(self.obj, stimVec)

    def sim(self, interval):
        lib.sim(self.obj, interval)

    def dumpSpikeStates(self):
        lib.dumpSpikeStates(self.obj)

    def set_t(self, t):
        lib.set_t(self.obj, t)

    def set_Ip(self, Ip):
        lib.set_Ip(self.obj, Ip)

    def set_STDP(self, _STDP):
        lib.set_STDP(self.obj, _STDP)

    def get_STDP(self):
        return lib.get_STDP(self.obj)

    def set_symmetric(self, _symmetric):
        lib.set_symmetric(self.obj, _symmetric)

    def get_symmetric(self):
        return lib.get_symmetric(self.obj)

    def set_homeostatic(self, _homeostatic):
        lib.set_homeostatic(self.obj, _homeostatic)

    def get_homeostatic(self):
        return lib.get_homeostatic(self.obj)

    def saveSpikes(self, saveflag):
        lib.saveSpikes(self.obj, saveflag)

    def perturbU(self):
        lib.perturbU(self.obj)

    def getUU(self):
        resp = lib.getUU(self.obj)
        return resp

    def getTheta(self):
        return lib.getTheta(self.obj)

    def setUU(self, UU):
        lib.setUU(self.obj, UU)

    def setStimIntensity(self, _StimIntensity):
        lib.setStimIntensity(self.obj, _StimIntensity)

    def calcMeanW(self, _iIDs, _jIDs):
        leniIDs, lenjIDs = len(_iIDs), len(_jIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))
        jIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]
        for j in range(lenjIDs):
            jIDs[j] = _jIDs[j]

        return lib.calcMeanW(self.obj, iIDs, leniIDs, jIDs, lenjIDs)

    def calcMeanF(self, _iIDs):
        leniIDs = len(_iIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]

        return lib.calcMeanF(self.obj, iIDs, leniIDs)

    def calcMeanD(self, _iIDs):
        leniIDs = len(_iIDs)
        iIDs = np.ascontiguousarray(np.zeros((self.NE,)).astype('int32'))

        for i in range(leniIDs):
            iIDs[i] = _iIDs[i]

        return lib.calcMeanD(self.obj, iIDs, leniIDs)

    def set_mex(self, mex):
        lib.set_mex(self.obj, mex)

    def set_hEhI(self, hE, hI):
        lib.set_hEhI(self.obj, hE, hI)

    def set_HAGA(self, _HAGA):
        lib.set_HAGA(self.obj, _HAGA)

    # NOTE: in progress
    def getRecents(self):
        resp = lib.getRecents(self.obj)
        return resp

    # def __setattr__(self, name, value):
    #     typ = [t for n, t in self.params_c_obj._fields_ if n == name][0]
    #     typstr = typ.__name__

    #     if typstr == 'c_int':
    #         self.params_c_obj.__setattr__(name, typ, c_int(value))
    #     # if the current field must be c_double
    #     elif typstr == 'c_double':
    #         self.params_c_obj.__setattr__(name, c_double(value))
    #     # if the current field must be c_bool
    #     elif typstr == 'c_bool':
    #         self.params_c_obj.__setattr__(name, c_bool(value))
    #     else:
    #         raise TypeError(f"Can't cast {name} to {typstr}.")

    #     lib.setParams(self.obj, self.params_c_obj)

    # def __getattr__(self, name):
    #     resp = lib.getState(self.obj)
    #     return getattr(resp, name)


if __name__ == "__main__":
    m = cClassOne(460, 92, 0, 6347)