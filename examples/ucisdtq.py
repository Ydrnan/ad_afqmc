from pyscf import gto, scf

from trot.afqmc import Afqmc
from trot.staging import stage_from_ccpy

try:
    from ccpy.drivers.driver import Driver
except Exception as exc:
    raise RuntimeError("This example requires ccpy. Please install ccpy to run it.") from exc

mol = gto.M(
    atom="H 0 0 0; H 0 0 1.1; H 0 1.7 0",
    basis="6-31g",
    symmetry="c1",
    charge=0,
    spin=1,
    verbose=3,
)

mf = scf.UHF(mol)
mf.max_cycle = 300
mf.run(conv_tol=1.0e-12)
if not mf.converged:
    raise RuntimeError("UHF did not converge.")

cc_driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
cc_driver.options["amp_convergence"] = 1.0e-12
cc_driver.options["energy_convergence"] = 1.0e-12
cc_driver.options["RHF_symmetry"] = False
cc_driver.run_cc(method="ccsdt")

staged = stage_from_ccpy(cc_driver, mf, order=4, chol_cut=1.0e-14, verbose=False)

af = Afqmc(staged)
af.walker_kind = "unrestricted"
af.n_walkers = 80
af.n_eql_blocks = 5
af.n_blocks = 20
af.seed = 7
mean, err = af.kernel()
print(f"AFQMC/UCISDTQ energy: {mean:.10f} +/- {err:.10f} Ha")
