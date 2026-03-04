from ad_afqmc_prototype import config

config.configure_once()
from ad_afqmc_prototype.wrapper.lno_wrapper import LnoRhf
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from pyscf import gto, scf

from ad_afqmc_prototype import testing
from ad_afqmc_prototype.afqmc import AFQMC
from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.trial.rhf import RhfTrial
from ad_afqmc_prototype.wrapper.lno_wrapper import run_afqmc_lno_mf

def _make_random_rhf_trial(key, norb, nocc):
    return RhfTrial(mo_coeff=testing.rand_orthonormal_cols(key, norb, nocc))


def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf

mf = mf()  # type: ignore


@pytest.mark.parametrize(
    "mf, e_ref, err_ref,norb_frozen",
    [
        (mf,-0.06714345  ,0.01170026,0),
        (mf,-0.04946941 , 0.00817172,1),
    ],
)
def test_calc_rhf(mf, params, e_ref, err_ref,norb_frozen):
    mo_coeff = mf.mo_coeff
    orbs_frozen = [i for i in range(norb_frozen)]
    norb_act = mf.mo_coeff.shape[1] - norb_frozen
    nactocc = mf.mol.nelectron // 2 - norb_frozen
    prjlo = [[0] for i in range(nactocc-1)] + [[1]]

    elcorr_afqmc, err_afqmc = run_afqmc_lno_mf(
    mf,
    mo_coeff=mo_coeff,
    norb_act=norb_act,
    nelec_act=nactocc * 2,
    norb_frozen=orbs_frozen,
    n_walkers=params.n_walkers,
    nblocks=params.n_blocks,
    seed=params.seed,
    chol_cut=1e-5,
    target_error=0,
    dt=0.01,
    prjlo=prjlo,
    n_eql=params.n_eql_blocks,
    ) 

    assert jnp.isclose(elcorr_afqmc, e_ref), (elcorr_afqmc, e_ref, elcorr_afqmc - e_ref)
    assert jnp.isclose(err_afqmc, err_ref), (err_afqmc, err_ref, err_afqmc - err_ref)

@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
