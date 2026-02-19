from pyscf import gto, scf
from ad_afqmc_prototype.wrapper.rhf import Rhf

r = 1.0
mol = gto.M(
    atom=f"H 0 0 0; H 0 0 {1.0*r}; H 0 0 {2.0*r}; H 0 0 {3.0*r}",
    basis="sto-6g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

afqmc = Rhf(mf)
mean, err, block_e_all, block_w_all = afqmc.kernel()

