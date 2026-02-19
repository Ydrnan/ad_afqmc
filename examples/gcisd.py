from pyscf import cc, gto, scf

from ad_afqmc_prototype.afqmc import AFQMC

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
mf = scf.GHF(mol).newton()
mf.kernel()

mycc = cc.GCCSD(mf)
mycc.kernel()
et = mycc.ccsd_t()  # for comparison
print(f"CCSD(T) total energy: {mycc.e_tot + et}")

afqmc = AFQMC(mycc)
afqmc.n_walkers = 100  # number of walkers
afqmc.n_eql_blocks = 10  # number of equilibration blocks
afqmc.n_blocks = 100  # number of sampling blocks
mean, err = afqmc.kernel()
