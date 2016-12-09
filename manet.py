#!/usr/bin/env python
'''
MANET (MANchester Energy Test)
python version based on numpy

To see example of usage try ./manet.py -h
'''

import numpy as np
import time


def gaussian(r2, sigma=0.2):
    '''
    N.B. the firs argument is not the distance but already its square!
    '''
    return np.exp(- 0.5 * r2 / sigma**2)


def eucliDistsSq(array1, array2):
    '''
    given 2 array of arrays return a matrix 
    where each element is the square of
    euclidian distance between aa[i] and bb[j]
    i.e. cc[i,j] = ((aa[i]-bb[j])**2).sum()
    '''
    return ((array1[:,None]-array2)**2).sum(axis=2)


def getPsi(array1, array2, func=gaussian, *argv, **argk):
    '''
    Input: array1, array2, func=gaussian, *argv, **argk
    output: Big matrix with all the psi_ij distances with the func already applied
    for both the particles with particles, antiparticles with antiparticles and cross
    terms on the off-diagonals matrices, the diagonal is zero
    |             |             |
    | psi n1 x n1 | psi n1 x n2 |  
    |             |             |
    |-------------|-------------|
    | psi n2 x n1 | psi n2 x n2 |  
    |             |             |
    '''
    print 'Computing Psi'
    t0 = time.time()
    arrayTot = np.concatenate([array1, array2])
    # More direct method, for some reason much slower
    # Psi = func(eucliDistsSq(arrayTot, arrayTot), *argv, **argk)
    # Compute full matrix in steps
    aa = func(eucliDistsSq(array1, array1), *argv, **argk)
    bb = func(eucliDistsSq(array2, array2), *argv, **argk)
    ab = func(eucliDistsSq(array1, array2), *argv, **argk)
    Psi = np.concatenate([np.concatenate([aa,ab],axis=1), np.concatenate([ab.T,bb],axis=1)],axis=0)
    np.fill_diagonal(Psi, 0)
    print 'Time taken to compute Psi is {0:.2f} seconds'.format(time.time()-t0)
    return Psi

def vectorsTi_fromArrays(array1, array2, func=gaussian, *argv, **argk):
    '''
    input: array1, array2, func=gaussian, *argv, **argk
    output T1, T2 (arrays of Tis)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func
    '''
    Ti1, Ti2 = 0, 0
    n1 = array1.shape[0]
    n2 = array2.shape[0]

    # The first two sums
    time0 = time.time()
    matrDists = func(eucliDistsSq(array1, array1), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Ti1 = matrDists.sum(1)/(2*n1*(n1-1))
    matrDists = func(eucliDistsSq(array2, array2), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Ti2 = matrDists.sum(1)/(2*n2*(n2-1)) 

    # and the mixed term
    matrDists = func(eucliDistsSq(array1, array2), *argv, **argk)
    Ti1 -= matrDists.sum(1)/(2*n1*n2)
    Ti2 -= matrDists.sum(0)/(2*n1*n2)
    
    return Ti1, Ti2

def vectorsTi_fromPsi(Psi, tau1):
    '''
    input: Psi, tau1
    output T1, T2 (arrays of Tis)
    Psi is the matrix with already the function of the distances squares 
    computed and vector with 1 for element of the first sample and 0 for the second one
    See docstring of getPsi for more information.
    This should be faster for permutations.
    '''
    n1 = tau1.sum()
    n2 = len(tau1)-n1
    
    # Vectors identities
    # tau1 = np.concatenate([np.ones(n1), np.zeros(n2)])
    tau2 = 1-tau1
    
    # Compute Tis
    Ti1 = np.tensordot((tau1/(2*n1*(n1-1)) - tau2/(2*n1*n2)),Psi,1) * tau1
    Ti2 = np.tensordot(Psi,(tau2/(2*n2*(n2-1)) - tau1/(2*n1*n2)),1) * tau2
    
    return Ti1[Ti1!=0], Ti2[Ti2!=0]
 

def vectorsTi(array1=None, array2=None, Psi=None, tau1=None, func=gaussian, *argv, **argk):
    '''
    input: array1, array2, func=gaussian, *argv, **argk
    output T1, T2 (arrays of Tis)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func

    As an alternative instead of array1 and array2 one can pass
    Psi and tau1 (matrix with already the function of the distances squares 
    computed and vector with 1 for element of the first sample and 0 for the second one)
    See docstring of getPsi for more information.
    This should be faster for permutations.
    '''

    if type(array1) != type(None) and type(array2) != type(None) and type(Psi) == type(None) and type(tau1) == type(None):
        Ti1, Ti2 = vectorsTi_fromArrays(array1=array1, array2=array2, func=gaussian, *argv, **argk)
        
    elif type(array1) == type(None) and type(array2) == type(None) and type(Psi) != type(None) and type(tau1) != type(None):
        Ti1, Ti2 = vectorsTi_fromPsi(Psi, tau1)

    else:
        raise IOError('vectorsTi take exactly (array1 and array2) or (Psi and tau)')

    return Ti1, Ti2


def EnergyTest(*argv, **argk):
    '''
    input: array1, array2, func=gaussian, *argv, **argk
    output T (the energy test)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func
    '''
    T1, T2 = vectorsTi(*argv, **argk)
    return T1.sum() + T2.sum()


def getTandMinMaxTi(*argv, **argk):
    '''
    input: array1, array2, func=gaussian, *argv, **argk
    output T, T1, T2 (the energy test and the two vectors of Tis)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func
    '''
    T1, T2 = vectorsTi(*argv, **argk)
    return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())

# def getTandMinMaxTi_fromPsi(Psi, tau1):
#     T1, T2 = vectorsTi_fromPsi(Psi, tau1)
#     return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())


def getTsPermutations(Psi, tau1, nperm=1):
    '''
    Takes as an input Psi (matrices with function already computed) and number of permutations
    '''
    Ts = []
    for i in xrange(nperm):
        print 'Computing permutation', i
        t0 = time.time()
        tau_perm = np.random.permutation(tau1)
        T1, T2 = vectorsTi_fromPsi(Psi, tau_perm)
        T = T1.sum() + T2.sum()
        Ts.append([T, min(T1.min(), T2.min()), max(T1.max(), T2.max())])
        print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(T, time.time()-t0)
    return Ts


def permutation(array1, array2):
    '''
    Take 2 arrays with the datasamples and return 
    2 arrays of the same size of the ones given but 
    with the elemats mixed randomly
    '''
    permut = np.random.permutation(np.concatenate([array1, array2]))
    return permut[:array1.shape[0]], permut[array1.shape[0]:]


def writeTis(array1, array2, T1, T2, outFile_name='Tis.txt'):
    '''
    Write files file with 
    0/1 var1 var2 ... varN Ti
    0/1 tells which samples it comes from,
    var1.. varN are the phase-space variables and
    Ti is the local Ti that should allow to visualize 
    which region shows disagreement between the two samples
    '''
    table1 = np.hstack((np.zeros((array1.shape[0],1)),array1,T1[:,None]))
    table2 = np.hstack((np.ones((array2.shape[0],1)),array2,T2[:,None]))
    tableTot = np.concatenate((table1, table2))
    fmt = ['%.0f']+['%.5e']*(array1.shape[1]+1)
    np.savetxt(outFile_name, tableTot, fmt = fmt)


class Manet:
    '''
    Class to stear energy test, essentially just uses functions above
    but once it computed the energy test once it stores the result
    so one can call the various quantities without redoing it
    '''
    def __init__(self, array1, array2, func=gaussian, *argv, **argk):
        self.array1 = array1
        self.array2 = array2
        self.func = func
        self.argv = argv
        self.argk = argk
        self._T1 = None
        self._T2 = None
        self._T = None
        self._maxTi = None
        self._minTi = None
        self._Psi = None
        self._tau1 = np.concatenate([np.ones(self.array1.shape[0]),np.zeros(self.array2.shape[0])])
        self._Ts = None

    def _computePsi(self):
        self._Psi = getPsi(array1=self.array1, array2=self.array2, func=self.func, *self.argv, **self.argk)

    def _computeEnergyTest(self):
        print 'Computing Energy Test'
        if type(self._Psi) == type(None):
            self._T1, self._T2 = vectorsTi_fromArrays(array1=self.array1, array2=self.array2, func=self.func, *self.argv, **self.argk)
        else:
            self._T1, self._T2 = vectorsTi_fromPsi(Psi=self.Psi, tau1=self._tau1)
        self._T = self._T1.sum() + self._T2.sum()

    @property    
    def Psi(self):
        if type(self._Psi) == type(None):
            self._computePsi()
        return self._Psi
        
    @property    
    def T1(self):
        if type(self._T) == type(None):
           self._computeEnergyTest()
        return self._T1

    @property    
    def T2(self):
        if type(self._T) == type(None):
            self._computeEnergyTest()
        return self._T2

    @property    
    def T(self):
        if type(self._T) == type(None):
            self._computeEnergyTest()
        return self._T

    @property
    def maxTi(self):
        if type(self._maxTi) == type(None):
            self._maxTi = max(self.T1.max(), self.T2.max())
        return self._maxTi

    @property
    def minTi(self):
        if np.tensordot(self._minTi) == np.tensordot(None):
            self._minTi = min(self.T1.min(), self.T2.min())
        return self._minTi

    def writeTis(self, outFile_name='Tis.txt'):
        writeTis(array1=self.array1, array2=self.array2, T1=self.T1, T2=self.T2, outFile_name=outFile_name)


    def Ts(self, nperm):
        if nperm == 0:
            return []
        if self._Ts == None:
            self._Ts = getTsPermutations(Psi=self.Psi, tau1=self._tau1, nperm=nperm)
            return self._Ts
        elif len(self._Ts) > nperm:
            return self._Ts[:nperm]
        elif len(self._Ts) < nperm:
            self._Ts.append(getTsPermutations(Psi=self.Psi, tau1=self._tau1, nperm=nperm-len(self._Ts)))
            return self._Ts

        
if __name__ == '__main__':


    ###   Options parser   ###
    import argparse
    parser = argparse.ArgumentParser(description = 'Compute the energy test on two different samples')
    parser.add_argument('file1', help='File with first sample')
    parser.add_argument('file2', help='File with second sample')
    parser.add_argument('-s','--sigma',help='specify the sigma parameter of the Gaussian metric', default=0.2, type=float)
    parser.add_argument('-n','--nevts',help='limit number of events in each sample to number given', type=int)
    parser.add_argument('-p','--nperm',help='define number of permutations to run', default=0, type=int)
    parser.add_argument('-r','--seed',help='specify a seed for the random number generator', default=0, type=int)
    parser.add_argument('-d',help='skip the default T calculation (used when just adding permutations as a separate job)',action='store_true')
    parser.add_argument('--slow',help='do not store matrix dstances in memory but recompute it every time',action='store_true')
    parser.add_argument('-o','--outfile',help='output file name, do not specify the estension as it will create a Tis.txt and a Ts.txt', default='')
    args = parser.parse_args()
    ##########################

    # set seed
    np.random.seed(args.seed)

    # read input datasets
    sample1 = np.loadtxt(args.file1)
    sample2 = np.loadtxt(args.file2)
    if args.nevts != None:
        sample1 = sample1[:args.nevts]
        sample2 = sample2[:args.nevts]

    et=Manet(sample1, sample2, sigma=args.sigma)
    
    if not args.d:
        t0 = time.time()
        outTis_name = args.outfile+'.Tis.txt' if args.outfile else 'Tis.txt'
        if not args.slow and args.nperm:
            et.Psi # Compute Psi so speed up permutations later
        et.writeTis(outTis_name)
        print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(et.T, time.time()-t0)

    # Run permutations
    if args.slow:
        Ts = []
        for i in xrange(args.nperm):
            print 'Computing permutation', i
            t0 = time.time()
            a, b = permutation(sample1, sample2)
            T, minTi, maxTi = getTandMinMaxTi(a, b, sigma=args.sigma)
            Ts.append([T, minTi, maxTi])
            print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(T, time.time()-t0)
    else:
        Ts = et.Ts(args.nperm)
      
    header = '' if args.d else '{0:.5e}'.format(et.T)
    outTs_name = args.outfile+'.Ts.txt' if args.outfile else 'Ts.txt'
    np.savetxt(outTs_name, Ts, header=header, comments = '', fmt='%.5e')

    

    

    
