from pyscf import gto, scf

from ad_afqmc_prototype.afqmc import AFQMC

mol = gto.M(
    atom="""
    N 2.5 0.0 0.0
    N 0.0 0.0 0.0
    """,
    basis="6-31g",
    verbose=3,
)
mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

afqmc = AFQMC(mf)
mean, err = afqmc.kernel()
