from pyscf import gto, scf, fci, ci, cc
from ad_afqmc import afqmc
import pickle
import numpy as np

mol =  gto.M(atom ="""
    H  0.0 0.0 0.0
    H 1.75 0.0 0.0
    H  0.0 0.0 10000.0
    H 1.75 0.0 10000.0
    """,
    basis = 'sto-3g',
    spin=0,
    verbose = 3)

# RHF
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

af = afqmc.AFQMC(mycc)
af.chol_cut = 1e-12
af.dt = 0.005
af.n_eql = 0
af.n_ene_blocks = 1
af.n_sr_blocks = 5
af.n_blocks = 100
af.n_prop_steps = 50
af.n_walkers = 20
af.trial= "cisd_pt"
af.walker_type= "rhf"
af.norb_frozen=0
af.ene0= mycc.e_tot
af.seed = 5
af.kernel()

