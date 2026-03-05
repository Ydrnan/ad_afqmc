from pyscf import gto, scf

from ad_afqmc_prototype import config

config.configure_once()

from ad_afqmc_prototype.afqmc import AFQMC

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="6-31g",
    verbose=3,
)
mf = scf.RHF(mol)
mf.kernel()

afqmc = AFQMC(mf)
mean, err, block_e_all, block_w_all = afqmc.kernel()
