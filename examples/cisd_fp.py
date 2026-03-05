from pyscf import gto, scf, cc

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import AFQMC_fp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31G",
    verbose=3,
)

# RHF
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

af = AFQMC_fp(mycc)
af.n_blocks = 30
af.ene0 = mycc.e_tot
af.n_traj = 10
af.seed = 6
mean, err = af.kernel()
