from pyscf import gto, scf

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import Afqmc, AfqmcFp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="6-31g",
    verbose=3,
)

mf = scf.RHF(mol)
mf.kernel()

af = Afqmc(mf)
af.chol_cut = 1e-6  # Cholesky decomposition threshold
af.norb_frozen = 1  # freeze O 1s core
af.n_walkers = 20  # number of walkers
af.n_eql_blocks = 10  # number of equilibration blocks
af.n_blocks = 200  # number of sampling blocks
mean1, err1 = af.kernel()  # Not required, juts to show it leads to the exact same result
af.save_staged("h2o.h5")  # Staged in h2o_af.h5

af2 = Afqmc.from_staged("h2o.h5")  # New instance from h2o_af.h5
# af2.norb_frozen = 1  # Cannot be changed as it has been staged
# af2.chol_cut = 1e-6  # Cannot be changed as it has been staged

# Parameters are NOT staged
af2.n_walkers = 20
af2.n_eql_blocks = 10
af2.n_blocks = 200
mean2, err2 = af2.kernel()

assert abs(mean1 - mean2) < 1e-12
assert abs(err1 - err2) < 1e-12

af = AfqmcFp(mf)
af.chol_cut = 1e-6  # Cholesky decomposition threshold
af.norb_frozen = 1  # freeze O 1s core
af.n_walkers = 20  # number of walkers
af.n_blocks = 5  # number of sampling blocks
af.n_traj = 4
af.ene0 = mf.e_tot
mean1, err1 = af.kernel()  # Not required, juts to show it leads to the exact same result
af.save_staged("h2o.h5")  # Staged in h2o_af.h5

af2 = AfqmcFp.from_staged("h2o.h5")  # New instance from h2o_af.h5
# af2.norb_frozen = 1  # Cannot be changed as it has been staged
# af2.chol_cut = 1e-6  # Cannot be changed as it has been staged

# Parameters are NOT staged
af2.n_walkers = 20
af2.n_blocks = 5
af2.n_traj = 4
af2.ene0 = mf.e_tot
mean2, err2 = af2.kernel()
