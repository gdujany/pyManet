'''
Tests for manet

Run from manet's root folder with "py.test"
'''

import os, sys
this_files_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_files_dir,'..'))
from manet import *
import pickle
import numpy as np

dd = pickle.load(open(os.path.join(this_files_dir, 'dd.pkl'),'rb'))

def test_getPsi():
    assert (dd['Psi'] - getPsi(dd['array1'], dd['array2']) < 1e-10).all()

def test_getPsi_bkg():
    assert (dd['Psi_bkg'] - getPsi(dd['array1'], dd['array2'], background1=dd['bkg1'], background2=dd['bkg2']) < 1e-10).all()

def test_Tis_fromPsi():
    for Ti1, Ti2 in zip(dd['Tis'], vectorsTi_fromPsi(dd['Psi'], dd['tau1'])):
        assert (Ti1-Ti2<1e-10).all() 

def test_Tis_fromArrays():
    for Ti1, Ti2 in zip(dd['Tis'], vectorsTi_fromArrays(dd['array1'], dd['array2'])):
        assert (Ti1-Ti2<1e-10).all() 
        

def test_Tis_fromPsi_bkg():
    for Ti1, Ti2 in zip(dd['Tis_bkg'], vectorsTi_fromPsi(dd['Psi_bkg'], dd['tau1'], dd['btau1'], dd['purity1'], dd['purity2'])):
        assert (Ti1-Ti2<1e-10).all() 
        
def test_Tis_fromArrays_bkg():
    for Ti1, Ti2 in zip(dd['Tis_bkg'], vectorsTi_fromArrays(dd['array1'], dd['array2'], dd['bkg1'], dd['bkg2'], dd['purity1'], dd['purity2'])):
        assert (Ti1-Ti2<1e-10).all() 

def test_T_fromPsi():
    assert round(dd['T']*1e10) == round(EnergyTest(Psi=dd['Psi'], tau1=dd['tau1'])*1e10)

def test_T_fromArrays():
    assert round(dd['T']*1e10) == round(EnergyTest(array1=dd['array1'], array2=dd['array2'])*1e10)

def test_T_fromPsi_bkg():
    assert round(dd['T_bkg']*1e10) == round(EnergyTest(Psi=dd['Psi_bkg'], tau1=dd['tau1'], btau1=dd['btau1'], purity1=dd['purity1'], purity2=dd['purity2'])*1e10)

def test_T_fromArrays_bkg():
    assert round(dd['T_bkg']*1e10) == round(EnergyTest(array1=dd['array1'], array2=dd['array2'], background1 = dd['bkg1'], background2 = dd['bkg2'], purity1=dd['purity1'], purity2=dd['purity2'])*1e10)
