from ad_afqmc_prototype import config

config.configure_once()

import jax
import jax.numpy as jnp
import pytest
from pyscf import gto, scf, cc

from ad_afqmc_prototype import driver
from ad_afqmc_prototype.afqmc import AFQMC_fp
from ad_afqmc_prototype.prop.types import QmcParams

@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("unrestricted",  -55.5896739004, 4.9120152e-03),
    ],
)
def test_calc_uhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    myafqmc = AFQMC_fp(mf)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean[-1].real, e_ref), (mean[-1].real, e_ref, mean[-1].real - e_ref)
    assert jnp.isclose(err[-1].real, err_ref), (err[-1].real, err_ref, err[-1].real - err_ref)


@pytest.fixture(scope="module")
def mf():
    mol = gto.M(
        atom="""
        N                 -1.67119571   -1.44021737    0.00000000
        H                 -2.12619571   -0.65213425    0.00000000
        H                 -0.76119571   -1.44021737    0.00000000
        """,
        basis="6-31G",
        spin = 1 ,
    )
    mf = scf.UHF(mol)
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_blocks=10,
        n_prop_steps=50,
        seed=6,
        n_walkers=5,
        n_ene_blocks = 2,
        dt = 0.05,
        ene0 = -55.60298562645659,
    )


if __name__ == "__main__":
    pytest.main([__file__])

