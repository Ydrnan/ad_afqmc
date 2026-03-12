from pyscf import gto, scf

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import Afqmc

mol = gto.M(
    atom="""
    N        0.0000000000      0.0000000000      0.0000000000
    H        1.0225900000      0.0000000000      0.0000000000
    H       -0.2281193615      0.9968208791      0.0000000000
    """,
    basis="6-31g",
    spin=1,
    verbose=3,
)
mf = scf.GHF(mol)
mf.kernel()

mo1 = mf.stability()
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

afqmc = Afqmc(mf, chol_cut=1e-8)
afqmc.mixed_precision = False
afqmc.n_walkers = 100  # number of walkers
afqmc.n_eql_blocks = 10  # number of equilibration blocks
afqmc.n_blocks = 100  # number of sampling blocks
mean, err = afqmc.kernel()
