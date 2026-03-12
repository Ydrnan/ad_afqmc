from pyscf import gto, scf, cc

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import AfqmcFp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

# RHF
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

af = AfqmcFp(mycc)
af.dt = 0.1
af.n_prop_steps = 10
af.n_blocks = 5
af.ene0 = mycc.e_tot
af.n_traj = 10
af.seed = 6
af.n_walkers = 200
af.walker_kind = "restricted"
mean, err = af.kernel()
