#!/usr/bin/env python
'''
MANET (MANchester Energy Test)
python version based on numpy

To see example of usage try ./manet.py -h
'''

import numpy as np


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


def vectorsTi(array1, array2, func=gaussian, *argv, **argk):
    '''
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


def EnergyTest(*argv, **argk):
    T1, T2 = vectorsTi(*argv, **argk)
    return T1.sum() + T2.sum()


def getTandMinMaxTi(*argv, **argk):
    T1, T2 = vectorsTi(*argv, **argk)
    return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())


def permutation(array1, array2):
    '''
    Take 2 arrays with the datasamples and return 
    2 arrays of the same size of the ones given but 
    with the elemats mixed randomly
    '''
    permut = np.random.permutation(np.concatenate([array1, array2]))
    return permut[:array1.shape[0]], permut[array1.shape[0]:]


def writeTis(array1, array2, T1, T2, outFile_name='Tis.txt'):
    table1 = np.hstack((np.zeros((array1.shape[0],1)),array1,T1[:,None]))
    table2 = np.hstack((np.ones((array2.shape[0],1)),array2,T2[:,None]))
    tableTot = np.concatenate((table1, table2))
    fmt = ['%.0f']+['%.5e']*(array1.shape[1]+1)
    np.savetxt('Tis.txt', tableTot, fmt = fmt)


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

    def _computeEnergyTest(self):
        print 'Computing Energy Test'
        self._T1, self._T2 = vectorsTi(self.array1, self.array2, self.func, *self.argv, **self.argk)
        self._T = self._T1.sum() + self._T2.sum()

    @property    
    def T1(self):
        if self._T == None:
           self._computeEnergyTest()
        return self._T1

    @property    
    def T2(self):
        if self._T == None:
            self._computeEnergyTest()
        return self._T2

    @property    
    def T(self):
        if self._T == None:
            self._computeEnergyTest()
        return self._T

    @property
    def maxTi(self):
        if self._maxTi == None:
            self._maxTi = max(self.T1.max(), self.T2.max())
        return self._maxTi

    @property
    def minTi(self):
        if self._minTi == None:
            self._minTi = min(self.T1.min(), self.T2.min())
        return self._minTi

    def writeTis(self, outFile_name='Tis.txt'):
        writeTis(array1=self.array1, array2=self.array2, T1=self.T1, T2=self.T2, outFile_name=outFile_name)

        
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

    import time

    if not args.d:
        t0 = time.time()
        et=Manet(sample1, sample2, sigma=args.sigma)
        et.writeTis()
        print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(et.T, time.time()-t0)

    # Run permutations
    Ts = []
    for i in xrange(args.nperm):
        print 'Computing permutation', i
        t0 = time.time()
        a, b = permutation(sample1, sample2)
        T, minTi, maxTi = getTandMinMaxTi(a, b, sigma=args.sigma)
        Ts.append([T, minTi, maxTi])
        print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(T, time.time()-t0)
      
    header = '' if args.d else '{0:.5e}'.format(et.T)
    np.savetxt('Ts.txt', Ts, header=header, comments = '', fmt='%.5e')

    

    

    
