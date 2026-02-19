from pyscf import gto, scf
from ad_afqmc_prototype.wrapper.rhf_fp import Rhf_fp as Rhf

r = 1.0
mol = gto.M(
    atom=f"H 0 0 0; H 0 0 {1.0*r}; H 0 0 {2.0*r}; H 0 0 {3.0*r}",
#atom ="""
#    O        0.0000000000      0.0000000000      0.0000000000
#    H        0.9562300000      0.0000000000      0.0000000000
#    H       -0.2353791634      0.9268076728      0.0000000000
#    """,
basis="sto-6g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

afqmc = Rhf(mf)
block_e_all, block_w_all = afqmc.kernel()

