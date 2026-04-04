from pyscf import cc, gto, scf

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

mycc = cc.GCCSD(mf)
mycc.kernel()
et = mycc.ccsd_t()  # for comparison
print(f"CCSD(T) total energy: {mycc.e_tot + et}")

af = Afqmc(mycc)
af.n_walkers = 100
af.n_eql_blocks = 10
af.n_blocks = 100
mean, err = af.kernel()
