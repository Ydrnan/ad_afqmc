from ad_afqmc_prototype import config

config.configure_once()

import jax.numpy as jnp
import pytest
from pyscf import gto, scf, cc

from ad_afqmc_prototype.afqmc import AFQMC_fp
from ad_afqmc_prototype.prop.types import QmcParamsFp


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -76.1196590973, 6.3008055e-04),
    ],
)
def test_calc_uhf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    myafqmc = AFQMC_fp(mycc)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean[-1].real, e_ref, atol=1e-6), (
        mean[-1].real,
        e_ref,
        mean[-1].real - e_ref,
    )
    assert jnp.isclose(err[-1].real, err_ref, atol=1e-6), (
        err[-1].real,
        err_ref,
        err[-1].real - err_ref,
    )


@pytest.fixture(scope="module")
def mycc():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="6-31G",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def params():
    return QmcParamsFp(
        n_blocks=10,
        n_prop_steps=50,
        seed=6,
        n_walkers=5,
        n_traj=2,
        dt=0.005,
        ene0=-76.11915086149004,
    )


if __name__ == "__main__":
    pytest.main([__file__])
