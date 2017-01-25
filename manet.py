#!/usr/bin/env python
'''
MANET (MANchester Energy Test)
python version based on numpy

To see example of usage try ./manet.py -h
'''

import numpy as np
import time


def gaussian(r2, sigma=10.0, mathLib=np):
    '''
    N.B. the first argument is not the distance but already its square!
    '''
    tmp = mathLib.exp(- 0.5 * r2 / sigma**2)
    # Set to zero in case distance is exactly zero
    tmp[tmp==1] = 0
    return tmp
    


def eucliDistsSq(array1, array2):
    '''
    given 2 array of arrays return a matrix 
    where each element is the square of
    euclidian distance between aa[i] and bb[j]
    i.e. cc[i,j] = ((aa[i]-bb[j])**2).sum()
    '''
    return ((array1[:,None]-array2)**2).sum(axis=2)


def getPsi(array1, array2, background1=None, background2=None, func=gaussian, *argv, **argk):
    '''
    Input: array1, array2, func=gaussian, *argv, **argk
    output: Big matrix with all the psi_ij distances with the func already applied
    for both the particles with particles, antiparticles with antiparticles and cross
    terms on the off-diagonals matrices, the diagonal is zero
    
    |             |             |
    | psi n1 x n1 | psi n1 x n2 |  
    |             |             |
    |-------------|-------------|
    |             |             |
    | psi n2 x n1 | psi n2 x n2 |  
    |             |             |
    
    If also the background is included then the matrix is 
    
    |             |             |             |             |
    | psi n1 x n1 | psi n1 x n2 | psi n1 x b1 | psi n1 x b2 |
    |             |             |             |             |
    |-------------|-------------|-------------|-------------|
    |             |             |             |             |
    | psi n2 x n1 | psi n2 x n2 | psi n2 x b1 | psi n2 x b2 |   
    |             |             |             |             |
    |-------------|-------------|-------------|-------------|
    |             |             |             |             |
    | psi b1 x n1 | psi b1 x n2 | psi b1 x b1 | psi b1 x b2 |
    |             |             |             |             |
    |-------------|-------------|-------------|-------------|
    |             |             |             |             |
    | psi b2 x n1 | psi b2 x n2 | psi b1 x b1 | psi b1 x b2 |   
    |             |             |             |             |
    |-------------|-------------|-------------|-------------|

    where b1 and b2 are the number of events in the two background samples
    
    '''
    print 'Computing Psi'
    t0 = time.time()
    # arrayTot = np.concatenate([array1, array2])
    # More direct method, for some reason much slower
    # Psi = func(eucliDistsSq(arrayTot, arrayTot), *argv, **argk)
    # Compute full matrix in steps
    aa = func(eucliDistsSq(array1, array1), *argv, **argk) # n1 x n1
    ab = func(eucliDistsSq(array1, array2), *argv, **argk) # n1 x n2
    bb = func(eucliDistsSq(array2, array2), *argv, **argk) # n2 x n2
    if type(background1) != type(None) and type(background2) != type(None):
        ac = func(eucliDistsSq(array1, background1), *argv, **argk) # n1 x b1
        ad = func(eucliDistsSq(array1, background2), *argv, **argk) # n1 x b2
        bc = func(eucliDistsSq(array2, background1), *argv, **argk) # n2 x b1
        bd = func(eucliDistsSq(array2, background2), *argv, **argk) # n2 x b2
        cc = func(eucliDistsSq(background1, background1), *argv, **argk) # b1 x b1 
        cd = func(eucliDistsSq(background1, background2), *argv, **argk) # b1 x b2
        dd = func(eucliDistsSq(background2, background2), *argv, **argk) # b2 x b2
        Psi = np.concatenate([np.concatenate([aa,ab,ac,ad],axis=1), np.concatenate([ab.T,bb,bc,bd],axis=1),
                              np.concatenate([ac.T,bc.T,cc,cd],axis=1),np.concatenate([ad.T,bd.T,cd.T,dd],axis=1)],axis=0)
    else:
        Psi = np.concatenate([np.concatenate([aa,ab],axis=1), np.concatenate([ab.T,bb],axis=1)],axis=0)
    np.fill_diagonal(Psi, 0)
    print 'Time taken to compute Psi is {0:.2f} seconds'.format(time.time()-t0)
    return Psi


def vectorsTi_fromArrays(array1, array2, background1=None, background2=None, purity1=None, purity2=None, bigData=False, func=gaussian, *argv, **argk):
    '''
    input: array1, array2,  background1, background2, purity1, purity2, func=gaussian, *argv, **argk
    output T1, T2, Tb1, Tb2 (arrays of Tis)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func
    if given background1 and background2 are the arrays with the background samples
    and purity1 and purity2 are the number of signal events in the array1 and array2 (absolute number, not ratio to total)
    If bigData is true use Dask to avoid to fill the RAM
    '''
    Ti1, Ti2 = 0,0
    n1 = 1.0*array1.shape[0]
    n2 = 1.0*array2.shape[0]

    if type(background1) != type(None) and type(background2) != type(None):
        if purity1 == None or purity2 == None:
            raise ValueError('If background samples are provided also purities should be given')
        w1, w2 = purity1, purity2
    else:
        w1, w2 = n1, n2

    if bigData:
        import dask.array as da
        array1 = da.from_array(array1, 1000)
        array2 = da.from_array(array2, 1000)
        mathLib = da
    else:
        mathLib = np
        
    # The first two sums
    time0 = time.time()
    matrDists = func(eucliDistsSq(array1, array1), mathLib=mathLib, *argv, **argk)
    #np.fill_diagonal(matrDists, 0) 
    Ti1 = matrDists.sum(1)/(2*w1*(w1-1))  #signal-signal component
    matrDists = func(eucliDistsSq(array2, array2), mathLib=mathLib, *argv, **argk)
    #np.fill_diagonal(matrDists, 0) 
    Ti2 = matrDists.sum(1)/(2*w2*(w2-1))  #signal-signal component
    
    # and the mixed term
    matrDists = func(eucliDistsSq(array1, array2), mathLib=mathLib, *argv, **argk)
    Ti1 -= matrDists.sum(1)/(2*w1*w2)
    Ti2 -= matrDists.sum(0)/(2*w1*w2)

    if type(background1) != type(None) and type(background2) != type(None):
        
        Tib1, Tib2 = 0,0
        b1 = background1.shape[0]
        b2 = background2.shape[0]
        bg1 = n1-w1
        bg2 = n2-w2
        if bigData:
            background1 = da.from_array(background1, 1000)
            background2 = da.from_array(background2, 1000)

        # The contribution of the background to first two sums
        matrDists = func(eucliDistsSq(array1, background1), mathLib=mathLib, *argv, **argk)
        Ti1 -= (bg1/b1)*(matrDists.sum(1)/(w1*(w1-1))) #subtract off the background-signal component in first term #note loss of 2 in denominator
        matrDists = func(eucliDistsSq(array2, background2), mathLib=mathLib, *argv, **argk)
        Ti2 -= (bg2/b2)*matrDists.sum(1)/(w2*(w2-1)) #subtract off the background-signal component in first term #note loss of 2 in denominator

        # and contribution of the background the mixed term
        matrDists = func(eucliDistsSq(array1, background2), mathLib=mathLib, *argv, **argk)
        Ti1 += (bg2/b2)*matrDists.sum(1)/(w1*w2)
        matrDists = func(eucliDistsSq(background1, array2), mathLib=mathLib, *argv, **argk)
        Ti2 += (bg1/b1)*matrDists.sum(0)/(w1*w2)
    
        #but we've now double counted, and removed the background-background term twice
        matrDists = func(eucliDistsSq(background1, background1), mathLib=mathLib, *argv, **argk)
        #np.fill_diagonal(matrDists, 0) 
        Tib1 =  ((bg1*bg1+bg1)/(b1*(b1-1)))* matrDists.sum(1)/(2*w1*(w1-1))
        matrDists = func(eucliDistsSq(background2, background2), mathLib=mathLib, *argv, **argk)
        #np.fill_diagonal(matrDists, 0) 
        Tib2 =  ((bg2*bg2+bg2)/(b2*(b2-1)))* matrDists.sum(1)/(2*w2*(w2-1))
        matrDists = func(eucliDistsSq(background1, background2), mathLib=mathLib, *argv, **argk)
        Tib1 -= (bg1/b1)*(bg2/b2)*matrDists.sum(1)/(2*w1*w2)
        Tib2 -= (bg1/b1)*(bg2/b2)*matrDists.sum(0)/(2*w1*w2)

        if bigData:
            return Ti1.compute(), Ti2.compute(), Tib1.compute(), Tib2.compute()
        else:
            return Ti1, Ti2, Tib1, Tib2

    if bigData:
        return Ti1.compute(), Ti2.compute()
    else:
        return Ti1, Ti2

def vectorsTi_fromPsi(Psi, tau1, btau1=None, purity1=None, purity2=None):
    '''
    input: Psi, tau1
    output T1, T2 (arrays of Tis)
    Psi is the matrix with already the function of the distances squares 
    computed and tau vector with 1 for element of the first sample and 0 for the second one.
    btau is like tau but for background samples
    See docstring of getPsi for more information.
    This should be faster for permutations.
    '''
    n1 = tau1.sum()
    n2 = (len(tau1)-n1)

    tau2 = 1-tau1

    if type(btau1) != type(None):

        # number events in background samples
        b1 = btau1.sum()
        b2 = (len(btau1) - b1)

        if purity1 == None or purity2 == None:
            raise ValueError('If background samples are provided also purities should be given')
        
        w1 = purity1 
        w2 = purity2 
        
        # Background yield in the sample
        bg1 = n1-w1
        bg2 = n2-w2
    
        btau2 = 1-btau1
        ttau1 = np.concatenate([tau1,np.zeros(len(btau1))])
        ttau2 = np.concatenate([tau2,np.zeros(len(btau1))])
        tbtau1 = np.concatenate([np.zeros(len(tau1)),btau1])
        tbtau2 = np.concatenate([np.zeros(len(tau1)),btau2])
    
        # Compute Tis
        Ti1 = np.tensordot(    (ttau1/(2*w1*(w1-1)) - ttau2/(2*w1*w2) - (bg1/b1)* tbtau1/(w1*(w1-1)) +(bg2/b2)*tbtau2/(w1*w2)),Psi,1) * ttau1
        Ti2 = np.tensordot(Psi,(ttau2/(2*w2*(w2-1)) - ttau1/(2*w1*w2) - (bg2/b2)* tbtau2/(w2*(w2-1)) +(bg1/b1)*tbtau1/(w1*w2))    ,1) * ttau2
        Tib1 = np.tensordot( ((bg1+1)/(b1-1))*(bg1/b1)*tbtau1/(2*w1*(w1-1)) - (bg1/b1)*(bg2/b2)*tbtau2/(2*w1*w2),Psi,1)*tbtau1
        Tib2 = np.tensordot(Psi, ((bg2+1)/(b2-1))*(bg2/b2)*tbtau2/(2*w2*(w2-1)) - (bg1/b1)*(bg2/b2)*tbtau1/(2*w1*w2),1)*tbtau2
        return Ti1[Ti1!=0], Ti2[Ti2!=0], Tib1[Tib1!=0], Tib2[Tib2!=0]

    else: # Case with no background
        # Compute Tis
        Ti1 = np.tensordot((tau1/(2*n1*(n1-1)) - tau2/(2*n1*n2)),Psi,1) * tau1
        Ti2 = np.tensordot(Psi,(tau2/(2*n2*(n2-1)) - tau1/(2*n1*n2)),1) * tau2
        return Ti1[Ti1!=0], Ti2[Ti2!=0]

    
def vectorsTi(array1=None, array2=None, background1 = None, background2 = None, Psi=None, tau1=None, btau1 = None, purity1=None, purity2=None, func=gaussian, bigData = False, *argv, **argk):
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
    
    if (type(array1) != type(None) and type(array2) != type(None)) and (type(Psi) == type(None) and type(tau1) == type(None)):
        return vectorsTi_fromArrays(array1=array1, array2=array2, background1=background1, background2 = background2, purity1=purity1, purity2=purity2, func=gaussian, bigData=bigData, *argv, **argk)
        
    elif (type(array1) == type(None) and type(array2) == type(None)) and (type(Psi) != type(None) and type(tau1) != type(None)):
        if bigData:
            raise IOError('If your dataset is large do not compute Psi')
        return vectorsTi_fromPsi(Psi, tau1, btau1, purity1, purity2)

    else:
        raise IOError('vectorsTi take exactly (array1 and array2) or (Psi and tau)')


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
    Tis = vectorsTi(*argv, **argk)
    return sum([Ti.sum() for Ti in Tis])


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
    T1, T2 = vectorsTi(*argv, **argk)[:2]

    return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())


def getTsPermutations(Psi, tau1, btau1=None, overallpurity=None, nperm=1):
    '''
    Takes as an input Psi (matrices with function already computed) and number of permutations
    '''
    Ts = []
    for i in xrange(nperm):
        print 'Computing permutation', i
        t0 = time.time()
        tau_perm = np.random.permutation(tau1)
        btau_perm = np.random.permutation(btau1) if type(btau1)!=type(None) else None
        samplesize = len(tau1)
        sample1size = tau1.sum()
        sample2size = samplesize - sample1size

        if type(btau1)!=type(None) and overallpurity == None:
            overallpurity = samplesize - len(btau1)

        #now to know the background fraction.
        newpurity1 = sample1size * (1.0*overallpurity) / (1.0*samplesize) if type(btau1)!=type(None) else None
        newpurity2 = sample2size * (1.0*overallpurity) / (1.0*samplesize) if type(btau1)!=type(None) else None
        
        Tis = vectorsTi_fromPsi(Psi, tau_perm, btau_perm, newpurity1, newpurity2)
        T = sum([Ti.sum() for Ti in Tis])
        Ts.append([T, min(Tis[0].min(), Tis[1].min()), max(Tis[0].max(), Tis[1].max())])
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
    but once it computes the energy test once it stores the result
    so one can call the various quantities without redoing it
    '''
    def __init__(self, array1, array2, background1=None, background2=None, func=gaussian, purity1=None, purity2=None, bigData=False, *argv, **argk):
        self.array1 = array1
        self.array2 = array2
        self.background1 = background1
        self.background2 = background2
        self.purity1 = purity1
        self.purity2 = purity2
        self.func = func
        self.bigData = bigData
        self.argv = argv
        self.argk = argk
        self._T1 = None
        self._T2 = None
        self._Tb1 = None
        self._Tb2 = None
        self._T = None
        self._maxTi = None
        self._minTi = None
        self._Psi = None
        self._tau1 = np.concatenate([np.ones(self.array1.shape[0]),np.zeros(self.array2.shape[0])])
        if type(self.background1) != type(None) and type(self.background2) != type(None):
            self._btau1 = np.concatenate([np.ones(self.background1.shape[0]),np.zeros(self.background2.shape[0])])
        else:
            self._btau1 = None
        self._Ts = None

    def _computePsi(self):
        self._Psi = getPsi(array1=self.array1, array2=self.array2,background1=self.background1, background2=self.background2, func=self.func, *self.argv, **self.argk)

        
    def _computeEnergyTest(self):
        print 'Computing Energy Test'
        if type(self._Psi) == type(None):
            Tis  = vectorsTi_fromArrays(array1=self.array1, array2=self.array2, background1=self.background1, background2 = self.background2, purity1=self.purity1, purity2=self.purity2, func=self.func, bigData=self.bigData, *self.argv, **self.argk)
        else:
            Tis = vectorsTi_fromPsi(Psi=self.Psi, tau1=self._tau1,btau1=self._btau1, purity1 = self.purity1, purity2 = self.purity2)
        self._T = sum([Ti.sum() for Ti in Tis]) 
        self._T1, self._T2 = Tis[:2]
        if len(Tis) == 4:
            self._Tb1, self._Tb2 = Tis[2:]

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
    def Tb1(self):
        if type(self._T) == type(None):
           self._computeEnergyTest()
        return self._Tb1

    @property    
    def Tb2(self):
        if type(self._T) == type(None):
            self._computeEnergyTest()
        return self._Tb2

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


    def Ts(self, overallpurity, nperm):
        if nperm == 0:
            return []
        if self._Ts == None:
            self._Ts = getTsPermutations(Psi=self.Psi, tau1=self._tau1,btau1=self._btau1, overallpurity=overallpurity, nperm=nperm)
            return self._Ts
        elif len(self._Ts) > nperm:
            return self._Ts[:nperm]
        elif len(self._Ts) < nperm:
            self._Ts.append(getTsPermutations(Psi=self.Psi, tau1=self._tau1, btau1=self._btau1, overallpurity=overallpurity, nperm=nperm-len(self._Ts)))
            return self._Ts

        
if __name__ == '__main__':

    ###   Options parser   ###
    import argparse
    parser = argparse.ArgumentParser(description = 'Compute the energy test on two different samples')
    parser.add_argument('file1', help='File with first sample')
    parser.add_argument('file2', help='File with second sample')
    parser.add_argument('-b', '--filesbackground', help='Two files with background samples (file1, file2)', nargs=2)
    parser.add_argument('-s','--sigma',help='specify the sigma parameter of the Gaussian metric', default=10.0, type=float)
    parser.add_argument('-e','--signals',help='''Number of signal events in first and second sample (s1, s2), 
    it is needed if background files are provided''', type=int, nargs=2)
    parser.add_argument('-n','--nevts',help='limit number of events in each sample to number given', type=float)
    parser.add_argument('-p','--nperm',help='define number of permutations to run', default=0, type=int)
    parser.add_argument('-r','--seed',help='specify a seed for the random number generator', default=0, type=int)
    parser.add_argument('-d',help='skip the default T calculation (used when just adding permutations as a separate job)',action='store_true')
    parser.add_argument('--slow',help='do not store matrix dstances in memory but recompute it every time',action='store_true')
    parser.add_argument('--bigData',help='Use Dask to avoid filling up the RAM, option slow is implicit',action='store_true')
    parser.add_argument('-o','--outfile',help='output file name, do not specify the estension as it will create a Tis.txt and a Ts.txt', default='')
    args = parser.parse_args()
    if args.bigData:
        args.slow = True
    ##########################

    # set seed
    print 'Using random number seed', args.seed
    np.random.seed(args.seed)

    # read input datasets
    sample1 = np.loadtxt(args.file1)
    sample2 = np.loadtxt(args.file2)
    if args.nevts != None:
        sample1 = sample1[:args.nevts]
        sample2 = sample2[:args.nevts]

    if args.filesbackground:    
        bkgsample1 = np.loadtxt(args.filesbackground[0])
        bkgsample2 = np.loadtxt(args.filesbackground[1])
        if args.signals == None:
            raise ValueError('If background samples are provided also purities should be given')
        else:
            purity1, purity2 = args.signals
    else:
        bkgsample1, bkgsample2, purity1, purity2 = [None]*4
        
    et=Manet(sample1, sample2, bkgsample1, bkgsample2, purity1=purity1 ,purity2=purity2, sigma=args.sigma, bigData=args.bigData)
    
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

            if args.filesbackground:  
                #also permute background.
                c, d = permutation(bkgsample1,bkgsample2)
                signalsize = (args.signals[0] + args.signals[1])
                perm1size = a.shape[0]
                perm2size = b.shape[0]
            
                newpurity1 = 1.0*signalsize * ((1.0*perm1size)/(1.0*(perm1size+perm2size)))
                newpurity2 = 1.0*signalsize * ((1.0*perm2size)/(1.0*(perm1size+perm2size)))
            else:
                c, d, newpurity1, newpurity2 = [None]*4
            T, minTi, maxTi = getTandMinMaxTi(a, b, c, d, purity1=newpurity1, purity2=newpurity2, sigma=args.sigma, bigData=args.bigData)
            Ts.append([T, minTi, maxTi])
            print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(T, time.time()-t0)
    else:
        overallsamplepurity = sum(args.signals) if args.filesbackground else None
        Ts = et.Ts(overallpurity = overallsamplepurity, nperm = args.nperm)
      
    header = '' if args.d else '{0:.5e}'.format(et.T)
    outTs_name = args.outfile+'.Ts.txt' if args.outfile else 'Ts.txt'
    np.savetxt(outTs_name, Ts, header=header, comments = '', fmt='%.5e')

    

    

    
