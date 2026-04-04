from ad_afqmc_prototype import config

config.configure_once()

import jax.numpy as jnp
import pytest
from pyscf import gto, scf

from ad_afqmc_prototype.afqmc import AfqmcFp
from ad_afqmc_prototype.prop.types import QmcParamsFp


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.7317723012, 2.9375393e-02),
        # ("unrestricted", -75.75594174131398, 0.01213379336719581),
    ],
)
def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    myafqmc = AfqmcFp(mf)
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
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def params():
    return QmcParamsFp(
        n_blocks=1,
        n_prop_steps=1000,
        seed=6,
        n_walkers=5,
        n_traj=3,
        dt=0.005,
        ene0=-75.67863248572299,
    )


if __name__ == "__main__":
    pytest.main([__file__])
