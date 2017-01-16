#!/usr/bin/env python
'''
Make dictionary with results to compare with tests
'''

import sys
sys.path.append('..')
from manet import *
import pickle

dd = {}

dd['array1'] = np.loadtxt('nocp1_mix.txt')[:5]
dd['array2'] = np.loadtxt('nocp1_mix.txt')[:5]

dd['bkg1'] = np.loadtxt('nocp1_bkgonly.txt')[:3]
dd['bkg2'] = np.loadtxt('nocp2_bkgonly.txt')[:3]

dd['purity1'], dd['purity2'] = 3, 3


dd['Psi'] = getPsi(dd['array1'], dd['array2'])
dd['Psi_bkg'] = getPsi(dd['array1'], dd['array2'], background1=dd['bkg1'], background2=dd['bkg2'])
dd['tau1'] = np.concatenate([np.ones(dd['array1'].shape[0]),np.zeros(dd['array2'].shape[0])])
dd['btau1'] = np.concatenate([np.ones(dd['bkg1'].shape[0]),np.zeros(dd['bkg2'].shape[0])])

dd['Tis'] = vectorsTi_fromPsi(dd['Psi'], dd['tau1'])
dd['Tis_bkg'] = vectorsTi_fromPsi(dd['Psi_bkg'], dd['tau1'], dd['btau1'], dd['purity1'], dd['purity2'])

dd['T'] = sum([Ti.sum() for Ti in dd['Tis']])
dd['T_bkg'] = sum([Ti.sum() for Ti in dd['Tis_bkg']])

pickle.dump(dd, open('dd.pkl','wb'))


