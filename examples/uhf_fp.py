from pyscf import gto, scf

from trot import config

config.configure_once(use_gpu=False)

from trot.afqmc import AfqmcFp

mol = gto.M(
    atom="""
    N  -1.67119571   -1.44021737    0.00000000
    H  -2.12619571   -0.65213425    0.00000000
    H  -0.76119571   -1.44021737    0.00000000
    """,
    spin=1,
    basis="6-31g",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

af = AfqmcFp(mf)
af.dt = 0.1
af.n_prop_steps = 100
af.n_blocks = 5
af.ene0 = mf.e_tot
af.n_traj = 10
af.seed = 6
af.n_walkers = 200
af.walker_kind = "unrestricted"
mean, err = af.kernel()
