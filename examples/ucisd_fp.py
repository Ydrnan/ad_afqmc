from pyscf import gto, scf, cc
from ad_afqmc_prototype.afqmc import AFQMC_fp

mol = gto.M(
    atom="""
    N                 -1.67119571   -1.44021737    0.00000000
    H                 -2.12619571   -0.65213425    0.00000000
    H                 -0.76119571   -1.44021737    0.00000000
             """,
    spin=1,
    basis="6-31G",
    verbose=3,
)

mf = scf.UHF(mol)
mf.kernel()

mycc = cc.UCCSD(mf)
mycc.kernel()

afqmc = AFQMC_fp(mycc)
afqmc.dt = 0.005
afqmc.n_blocks = 50
afqmc.ene0 = mycc.e_tot
afqmc.n_ene_blocks = 10
afqmc.seed = 6
afqmc.walker_kind = "unrestricted"
mean, err = afqmc.kernel()
