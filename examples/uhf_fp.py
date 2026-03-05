from pyscf import gto, scf

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import AFQMC_fp

mol = gto.M(
    atom="""
    O        0.0000000000      0.0000000000      0.0000000000
    H        0.9562300000      0.0000000000      0.0000000000
    H       -0.2353791634      0.9268076728      0.0000000000
    """,
    basis="cc-pvdz",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

afqmc = AFQMC_fp(mf)
afqmc.dt = 0.005
afqmc.n_blocks = 50
afqmc.ene0 = mf.e_tot
afqmc.n_traj = 10
afqmc.seed = 6
afqmc.n_walkers = 200
afqmc.walker_kind = "unrestricted"
mean, err = afqmc.kernel()
