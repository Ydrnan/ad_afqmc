from pyscf import gto, scf
from ad_afqmc_prototype.afqmc import AFQMC_fp

r = 1.0
mol = gto.M(
    atom=f"H 0 0 0; H 0 0 {1.0*r}; H 0 0 {2.0*r}; H 0 0 {3.0*r}",
    basis="sto-6g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

af = AFQMC_fp(mf)
af.dt = 0.005
af.n_blocks = 50
af.ene0 = mf.e_tot
af.n_ene_blocks = 10
af.seed = 6
mean, err = af.kernel()
