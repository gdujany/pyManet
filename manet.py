#!/usr/bin/env python
'''
MANET (MANchester Energy Test)
python version based on numpy

To see example of usage try ./manet.py -h
'''

import numpy as np
import time


def gaussian(r2, sigma=10.0):
    '''
    N.B. the first argument is not the distance but already its square!
    '''
    tmp = np.exp(- 0.5 * r2 / sigma**2)
    # Set to zero ..
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


def getPsi(array1, array2, background1, background2, func=gaussian, *argv, **argk):
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
    #arrayTot = np.concatenate([array1, array2])
    # More direct method, for some reason much slower
    # Psi = func(eucliDistsSq(arrayTot, arrayTot), *argv, **argk)
    # Compute full matrix in steps
    aa = func(eucliDistsSq(array1, array1), *argv, **argk)
    ab = func(eucliDistsSq(array1, array2), *argv, **argk)
    ac = func(eucliDistsSq(array1, background1), *argv, **argk)
    ad = func(eucliDistsSq(array1, background2), *argv, **argk)
    bb = func(eucliDistsSq(array2, array2), *argv, **argk)
    bc = func(eucliDistsSq(array2, background1), *argv, **argk)
    bd = func(eucliDistsSq(array2, background2), *argv, **argk)
    cc = func(eucliDistsSq(background1, background1), *argv, **argk)
    cd = func(eucliDistsSq(background1, background2), *argv, **argk)
    dd = func(eucliDistsSq(background2, background2), *argv, **argk)
    Psi = np.concatenate([np.concatenate([aa,ab,ac,ad],axis=1), np.concatenate([ab.T,bb,bc,bd],axis=1),np.concatenate([ac.T,bc.T,cc,cd],axis=1),np.concatenate([ad.T,bd.T,cd.T,dd],axis=1)],axis=0)
    np.fill_diagonal(Psi, 0)
    print 'Time taken to compute Psi is {0:.2f} seconds'.format(time.time()-t0)
    return Psi

def vectorsTi_fromArrays(array1, array2, background1, background2, purity1, purity2, func=gaussian, *argv, **argk):
    '''
    input: array1, array2,  background1, background2, purity1, purity2, func=gaussian, *argv, **argk
    output T1, T2, Tb1, Tb2 (arrays of Tis)
    array1 and array2 are the arrays with the data to compare
    they are 2D numpy array where each row is a candidate and
    contains the PS variables eg [m12, m23, m13]
    func is the distance function 
    argv and argk are passed to func
    '''
    Ti1, Ti2, Tib1, Tib2 = 0, 0,0,0
    n1 = 1.0*array1.shape[0]
    n2 = 1.0*array2.shape[0]
    
    
    b1 = background1.shape[0]
    b2 = background2.shape[0]

    w1 = purity1
    w2 = purity2
    if purity1 < 0:
        w1 =  n1-b1 #we can set w1 to be the actual number of signal events.
        w2 =  n2-b2 #we can set w2 to be the actual number of signal events.
    bg1 = n1-w1
    bg2 = n2-w2

    print n1, n2, bg1, bg2, b1, b2, w1, w2
    
    # The first two sums
    time0 = time.time()
    matrDists = func(eucliDistsSq(array1, array1), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Ti1 = matrDists.sum(1)/(2*w1*(w1-1))  #signal-signal component
    matrDists = func(eucliDistsSq(array2, array2), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Ti2 = matrDists.sum(1)/(2*w2*(w2-1))  #signal-signal component
    matrDists = func(eucliDistsSq(array1, background1), *argv, **argk)
    Ti1 -= (bg1/b1)*(matrDists.sum(1)/(w1*(w1-1))) #subtract off the background-signal component in first term #note loss of 2 in denominator
    matrDists = func(eucliDistsSq(array2, background2), *argv, **argk)
    Ti2 -= (bg2/b2)*matrDists.sum(1)/(w2*(w2-1)) #subtract off the background-signal component in first term #note loss of 2 in denominator
    
    # and the mixed term
    matrDists = func(eucliDistsSq(array1, array2), *argv, **argk)
    Ti1 -= matrDists.sum(1)/(2*w1*w2)
    Ti2 -= matrDists.sum(0)/(2*w1*w2)
    matrDists = func(eucliDistsSq(array1, background2), *argv, **argk)
    Ti1 += (bg2/b2)*matrDists.sum(1)/(w1*w2)
    matrDists = func(eucliDistsSq(background1, array2), *argv, **argk)
    Ti2 += (bg1/b1)*matrDists.sum(0)/(w1*w2)
    
    #but we've now double counted, and removed the background-background term twice
    
    matrDists = func(eucliDistsSq(background1, background1), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Tib1 =  ((bg1*bg1+bg1)/(b1*(b1-1)))* matrDists.sum(1)/(2*w1*(w1-1))
    matrDists = func(eucliDistsSq(background2, background2), *argv, **argk)
    np.fill_diagonal(matrDists, 0) 
    Tib2 =  ((bg2*bg2+bg2)/(b2*(b2-1)))* matrDists.sum(1)/(2*w2*(w2-1))
    matrDists = func(eucliDistsSq(background1, background2), *argv, **argk)
    Tib1 -= (bg1/b1)*(bg2/b2)*matrDists.sum(1)/(2*w1*w2)
    Tib2 -= (bg1/b1)*(bg2/b2)*matrDists.sum(0)/(2*w1*w2) 
    
    print 'hurrah'
    print Ti1.sum(), Ti2.sum(), Tib1.sum(), Tib2.sum()
    return Ti1, Ti2, Tib1, Tib2

def vectorsTi_fromPsi(Psi, tau1,btau1,purity1, purity2):
    '''
    input: Psi, tau1
    output T1, T2 (arrays of Tis)
    Psi is the matrix with already the function of the distances squares 
    computed and vector with 1 for element of the first sample and 0 for the second one
    See docstring of getPsi for more information.
    This should be faster for permutations.
    '''
    
    n1 = tau1.sum()
    n2 = (len(tau1)-n1)

    #number of background events we can estimate from
    b1 = btau1.sum()
    b2 = (len(btau1) - b1)

    #what about situation when we need to pass it... hmmm.
    w1 = purity1
    w2 = purity2
    if purity1 < 0:
        w1 =  n1-b1 #we can set w1 to be the actual number of signal events.
        w2 =  n2-b2 #we can set w2 to be the actual number of signal events.

    #best estimate of background yield in the sample
    bg1 = n1-w1
    bg2 = n2-w2
    
    print n1, n2, b1, b2, w1, w2, bg1, bg2
    print bg1/b1, bg2/b2
    #need to create new vectors that are longer.
    
    # Vectors identities
    #tau1 =  np.concatenate([np.ones(n1), np.zeros(n2)])
    tau2 = 1-tau1
    
    #btau1 =  np.concatenate([np.ones(b1), np.zeros(b2)])
    btau2 = 1-btau1

    ttau1 = np.concatenate([tau1,np.zeros(len(btau1))])
    ttau2 = np.concatenate([tau2,np.zeros(len(btau1))])
    tbtau1 = np.concatenate([np.zeros(len(tau1)),btau1])
    tbtau2 = np.concatenate([np.zeros(len(tau1)),btau2])
   
    print len(tau1), len(btau1), len(tau2), len(btau2), len(ttau1), len(ttau2), len(tbtau1), len(tbtau2)
    # Compute Tis
    Ti1 = np.tensordot(    (ttau1/(2*w1*(w1-1)) - ttau2/(2*w1*w2) - (bg1/b1)* tbtau1/(w1*(w1-1)) +(bg2/b2)*tbtau2/(w1*w2)),Psi,1) * ttau1
    Ti2 = np.tensordot(Psi,(ttau2/(2*w2*(w2-1)) - ttau1/(2*w1*w2) - (bg2/b2)* tbtau2/(w2*(w2-1)) +(bg1/b1)*tbtau1/(w1*w2))    ,1) * ttau2
    Tib1 = np.tensordot( ((bg1+1)/(b1-1))*(bg1/b1)*tbtau1/(2*w1*(w1-1)) - (bg1/b1)*(bg2/b2)*tbtau2/(2*w1*w2),Psi,1)*tbtau1
    Tib2 = np.tensordot(Psi, ((bg2+1)/(b2-1))*(bg2/b2)*tbtau2/(2*w2*(w2-1)) - (bg1/b1)*(bg2/b2)*tbtau1/(2*w1*w2),1)*tbtau2
    print 'yo'
    return Ti1[Ti1!=0], Ti2[Ti2!=0], Tib1, Tib2
 

def vectorsTi(array1=None, array2=None, background1 = None, background2 = None, Psi=None, tau1=None, btau1 = None, purity1=-1.0, purity2=-1.0, func=gaussian, *argv, **argk):
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
        Ti1, Ti2, Tib1, Tib2 = vectorsTi_fromArrays(array1=array1, array2=array2, background1=background1, background2 = background2, purity1=purity1, purity2=purity2, func=gaussian, *argv, **argk)
        
    elif (type(array1) == type(None) and type(array2) == type(None)) and (type(Psi) != type(None) and type(tau1) != type(None)):
        Ti1, Ti2, Tib1, Tib2 = vectorsTi_fromPsi(Psi, tau1, btau1, purity1, purity2)

    else:
        raise IOError('vectorsTi take exactly (array1 and array2) or (Psi and tau)')

    return Ti1, Ti2, Tib1, Tib2


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
    T1, T2, Tb1, Tb2 = vectorsTi(*argv, **argk)
    #print T1.sum(), T2.sum(), Tb1.sum(), Tb2.sum()
    return T1.sum() + T2.sum() + Tb1.sum() + Tb2.sum()


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
    T1, T2, Tb1, Tb2 = vectorsTi(*argv, **argk)
    #print T1.sum(), T2.sum(), Tb1.sum(), Tb2.sum()
    return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())

# def getTandMinMaxTi_fromPsi(Psi, tau1):
#     T1, T2 = vectorsTi_fromPsi(Psi, tau1)
#     return (T1.sum() + T2.sum()), min(T1.min(), T2.min()), max(T1.max(), T2.max())


def getTsPermutations(Psi, tau1, btau1, overallpurity, nperm=1):
    '''
    Takes as an input Psi (matrices with function already computed) and number of permutations
    '''
    Ts = []
    for i in xrange(nperm):
        print 'Computing permutation', i
        t0 = time.time()
        tau_perm = np.random.permutation(tau1)
        btau_perm = np.random.permutation(btau1)
        samplesize = len(tau1)
        sample1size = tau1.sum()
        sample2size = samplesize - sample1size
        #now to know the background fraction.
        newpurity1 = sample1size * (1.0*overallpurity) / (1.0*samplesize)
        newpurity2 = sample2size * (1.0*overallpurity) / (1.0*samplesize)
        print newpurity1, newpurity2 , sample1size, sample2size
        
        T1, T2, Tb1, Tb2 = vectorsTi_fromPsi(Psi, tau_perm, btau_perm,newpurity1, newpurity2 )
        T = T1.sum() + T2.sum() + Tb1.sum() + Tb2.sum() 
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
    def __init__(self, array1, array2, background1, background2, func=gaussian, purity1 =  -1.0, purity2 = -1.0, *argv, **argk):
        self.array1 = array1
        self.array2 = array2
        self.background1 = background1
        self.background2 = background2
        self.purity1 = purity1
        self.purity2 =purity2
        self.func = func
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
        self._btau1 = np.concatenate([np.ones(self.background1.shape[0]),np.zeros(self.background2.shape[0])])
        self._Ts = None

    def _computePsi(self):
        self._Psi = getPsi(array1=self.array1, array2=self.array2,background1=self.background1, background2=self.background2, func=self.func, *self.argv, **self.argk)

    def _computeEnergyTest(self):
        print 'Computing Energy Test'
        if type(self._Psi) == type(None):
            self._T1, self._T2, self._Tb1, self._Tb2 = vectorsTi_fromArrays(array1=self.array1, array2=self.array2, background1=self.background1, background2 = self.background2, purity1=self.purity1, purity2=self.purity2, func=self.func, *self.argv, **self.argk)
        else:
            self._T1, self._T2, self._Tb1, self._Tb2 = vectorsTi_fromPsi(Psi=self.Psi, tau1=self._tau1,btau1=self._btau1, purity1 = self.purity1, purity2 = self.purity2)
        self._T = self._T1.sum() + self._T2.sum() + self._Tb1.sum() + self._Tb2.sum()

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
            self._Ts = getTsPermutations(Psi=self.Psi, tau1=self._tau1,btau1=self._btau1, overallpurity = overallpurity, nperm=nperm)
            return self._Ts
        elif len(self._Ts) > nperm:
            return self._Ts[:nperm]
        elif len(self._Ts) < nperm:
            self._Ts.append(getTsPermutations(Psi=self.Psi, tau1=self._tau1, btau1=self._btau1,overallpurity=  overallpurity, nperm=nperm-len(self._Ts)))
            return self._Ts

        
if __name__ == '__main__':


    ###   Options parser   ###
    import argparse
    parser = argparse.ArgumentParser(description = 'Compute the energy test on two different samples')
    parser.add_argument('file1', help='File with first sample')
    parser.add_argument('file2', help='File with second sample')
    parser.add_argument('filebackground1', help='File with first background sample')
    parser.add_argument('filebackground2', help='File with second background sample')
    parser.add_argument('-s','--sigma',help='specify the sigma parameter of the Gaussian metric', default=10.0, type=float)
    parser.add_argument('-p1','--purity1',help='Number of singal events in first sample', default=1.0, type=int)
    parser.add_argument('-p2','--purity2',help='Number of signal events in second sample', default=1.0, type=int)
    parser.add_argument('-n','--nevts',help='limit number of events in each sample to number given', type=int)
    parser.add_argument('-p','--nperm',help='define number of permutations to run', default=0, type=int)
    parser.add_argument('-r','--seed',help='specify a seed for the random number generator', default=0, type=int)
    parser.add_argument('-d',help='skip the default T calculation (used when just adding permutations as a separate job)',action='store_true')
    parser.add_argument('--slow',help='do not store matrix dstances in memory but recompute it every time',action='store_true')
    parser.add_argument('-o','--outfile',help='output file name, do not specify the estension as it will create a Tis.txt and a Ts.txt', default='')
    args = parser.parse_args()
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

        
    bkgsample1 = np.loadtxt(args.filebackground1)
    bkgsample2 = np.loadtxt(args.filebackground2)
        
    et=Manet(sample1, sample2, bkgsample1, bkgsample2, purity1 = args.purity1 ,purity2 = args.purity2 , sigma=args.sigma)
    if args.purity1< 0:
         print 'Assuming sample 1 and sample 2 yields known exactly and accurately'
    else:
        print 'Assuming sample 1 and sample 2 contain %s and %s signal events respectively' % (args.purity1,args.purity2)
    if not args.d:
        t0 = time.time()
        outTis_name = args.outfile+'.Tis.txt' if args.outfile else 'Tis.txt'
        if not args.slow:
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
            #also permute background.
            c, d = permutation(bkgsample1,bkgsample2)
            signalsize = (args.purity1 + args.purity2)
            perm1size = a.shape[0]
            perm2size = b.shape[0]
            
            newpurity1 = 1.0*signalsize * ((1.0*perm1size)/(1.0*(perm1size+perm2size)))
            newpurity2 = 1.0*signalsize * ((1.0*perm2size)/(1.0*(perm1size+perm2size)))
            print "permutation sizes", signalsize, perm1size, perm2size, newpurity1, newpurity2
            #we now need to get a feel for how large a and b are.
            T, minTi, maxTi = getTandMinMaxTi(a, b, c, d, purity1 = newpurity1 ,purity2 = newpurity2,sigma=args.sigma)#note we're going to need to give new purity here - which will be about 50%
            Ts.append([T, minTi, maxTi])
            print 'T = {0:.5e}, time taken {1:.2f} seconds'.format(T, time.time()-t0)
    else:
        overallsamplepurity = args.purity1 + args.purity2
        Ts = et.Ts(overallpurity = overallsamplepurity, nperm = args.nperm)
      
    header = '' if args.d else '{0:.5e}'.format(et.T)
    outTs_name = args.outfile+'.Ts.txt' if args.outfile else 'Ts.txt'
    np.savetxt(outTs_name, Ts, header=header, comments = '', fmt='%.5e')

    

    

    
