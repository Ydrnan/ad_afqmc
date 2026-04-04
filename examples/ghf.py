from pyscf import gto, scf

from trot.afqmc import Afqmc

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

af = Afqmc(mf, chol_cut=1e-8)
af.mixed_precision = False
af.n_walkers = 100
af.n_eql_blocks = 10
af.n_blocks = 100
mean, err = af.kernel()
