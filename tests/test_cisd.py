from ad_afqmc_prototype import config

config.configure_once()

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from ad_afqmc_prototype import testing
from ad_afqmc_prototype.afqmc import Afqmc
from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.meas.cisd import (
    build_meas_ctx,
    energy_kernel_rw_rh,
    force_bias_kernel_rw_rh,
    make_cisd_meas_ops,
    rdm1_kernel_rw,
)

# from ad_afqmc_prototype.prep.pyscf_interface import get_cisd
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.trial.cisd import CisdTrial, make_cisd_trial_ops, overlap_r


def _make_cisd_trial(
    key,
    norb: int,
    nocc: int,
    *,
    memory_mode: Literal["low", "high"] = "low",
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
) -> CisdTrial:
    """
    Random CISD coefficients in the MO basis where the reference occupies [0..nocc-1].

    We keep coefficients modest in magnitude to reduce catastrophic cancellation
    when comparing against overlap-based finite differences.
    """
    nvir = norb - nocc
    k1, k2 = jax.random.split(key)

    ci1 = scale_ci1 * jax.random.normal(k1, (nocc, nvir), dtype=dtype)
    ci2 = scale_ci2 * jax.random.normal(k2, (nocc, nvir, nocc, nvir), dtype=dtype)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    # Use high precision for the "testing" dtypes so the manual kernel is not
    # artificially noisy from float32/complex64 paths.
    return CisdTrial(ci1=ci1, ci2=ci2)


def _pad_cisd_amplitudes(
    ci1: jax.Array,
    ci2: jax.Array,
    *,
    nocc_t_core: int,
    nvir_t_outer: int,
) -> tuple[jax.Array, jax.Array]:
    nocc_act, nvir_act = ci1.shape
    nocc_full = nocc_t_core + nocc_act
    nvir_full = nvir_act + nvir_t_outer

    ci1_full = jnp.zeros((nocc_full, nvir_full), dtype=ci1.dtype)
    ci1_full = ci1_full.at[nocc_t_core:, :nvir_act].set(ci1)

    ci2_full = jnp.zeros((nocc_full, nvir_full, nocc_full, nvir_full), dtype=ci2.dtype)
    ci2_full = ci2_full.at[nocc_t_core:, :nvir_act, nocc_t_core:, :nvir_act].set(ci2)
    return ci1_full, ci2_full


@pytest.mark.parametrize("nocc_t_core,nvir_t_outer", [(0, 0), (1, 2)])
def test_active_space_matches_zero_padded_full_space(nocc_t_core, nvir_t_outer):
    key = jax.random.PRNGKey(321)
    k1, k2, k_ham, k_w = jax.random.split(key, 4)

    nocc_full = 4
    nvir_full = 6
    nocc_act = nocc_full - nocc_t_core
    nvir_act = nvir_full - nvir_t_outer

    ci1 = 0.05 * jax.random.normal(k1, (nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.02 * jax.random.normal(k2, (nocc_act, nvir_act, nocc_act, nvir_act), dtype=jnp.float64)
    ci2 = 0.5 * (ci2 + ci2.transpose(2, 3, 0, 1))

    trial_active = CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    ci1_full, ci2_full = _pad_cisd_amplitudes(
        ci1,
        ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )
    trial_full = CisdTrial(ci1=ci1_full, ci2=ci2_full)

    norb_full = nocc_full + nvir_full
    ham = testing.make_random_ham_chol(k_ham, norb=norb_full, n_chol=8, basis="restricted")
    ctx_active = build_meas_ctx(ham, trial_active)
    ctx_full = build_meas_ctx(ham, trial_full)

    for i in range(4):
        walker = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb_full, nocc_full, mix=0.25
        )
        o_active = overlap_r(walker, trial_active)
        o_full = overlap_r(walker, trial_full)
        assert jnp.allclose(o_active, o_full, rtol=1e-12, atol=1e-12), (o_active, o_full)

        fb_active = force_bias_kernel_rw_rh(walker, ham, ctx_active, trial_active)
        fb_full = force_bias_kernel_rw_rh(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(fb_active, fb_full, rtol=1e-12, atol=1e-12), (fb_active, fb_full)

        e_active = energy_kernel_rw_rh(walker, ham, ctx_active, trial_active)
        e_full = energy_kernel_rw_rh(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(e_active, e_full, rtol=1e-12, atol=1e-12), (e_active, e_full)

        dm_active = rdm1_kernel_rw(walker, ham, ctx_active, trial_active)
        dm_full = rdm1_kernel_rw(walker, ham, ctx_full, trial_full)
        assert jnp.allclose(dm_active, dm_full, rtol=1e-12, atol=1e-12), (dm_active, dm_full)


@pytest.mark.parametrize("norb,nocc,n_chol,memory_mode", [(8, 3, 10, "low"), (10, 4, 12, "high")])
def test_auto_force_bias_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(123)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
            memory_mode=memory_mode,
        ),
        make_trial_ops_fn=make_cisd_trial_ops,
        make_meas_ops_fn=make_cisd_meas_ops,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # CISD overlap is more structured than RHF; auto (finite-diff / overlap-derivative)
        # can need a slightly looser tolerance.
        assert jnp.allclose(v_a, v_m, rtol=2e-5, atol=2e-6), (v_a, v_m)


@pytest.mark.parametrize("norb,nocc,n_chol,memory_mode", [(8, 3, 10, "low"), (10, 4, 12, "high")])
def test_auto_energy_matches_manual_cisd(norb, nocc, n_chol, memory_mode):
    walker_kind = "restricted"
    key = jax.random.PRNGKey(456)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nocc, nocc),
        n_chol,
        make_trial_fn=_make_cisd_trial,
        make_trial_fn_kwargs=dict(
            norb=norb,
            nocc=nocc,
            memory_mode=memory_mode,
        ),
        make_trial_ops_fn=make_cisd_trial_ops,
        make_meas_ops_fn=make_cisd_meas_ops,
    )

    if not meas_manual.has_kernel(k_energy):
        pytest.skip("manual CISD meas does not provide k_energy")

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing.make_restricted_walker_near_ref(
            jax.random.fold_in(k_w, i), norb, nocc, mix=0.25
        )

        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=2e-5, atol=2e-6), (ea, em)


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.72869718476204, 0.0002352938315467452),
    ],
)
def test_calc_rhf_hamiltonian(mycc, params, walker_kind, e_ref, err_ref):
    myafqmc = Afqmc(mycc)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()

    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


def test_stage_infers_trial_freeze_from_list_valued_cc_frozen(mycc_frozen_list):
    staged = Afqmc(mycc_frozen_list).stage()

    assert staged.ham.norb_frozen == 0
    assert staged.ham.norb == mycc_frozen_list._scf.mo_coeff.shape[-1]
    assert staged.trial.kind == "cisd"
    assert staged.trial.data["ci1"].shape == mycc_frozen_list.t1.shape
    assert int(staged.trial.data["nocc_t_core"].item()) == 1
    assert int(staged.trial.data["nvir_t_outer"].item()) == 1


def test_stage_splits_list_valued_cc_frozen_with_afqmc_frozen_core(mycc_frozen_list):
    staged = Afqmc(mycc_frozen_list, norb_frozen=1).stage()

    assert staged.ham.norb_frozen == 1
    assert staged.ham.norb == mycc_frozen_list._scf.mo_coeff.shape[-1] - 1
    assert staged.trial.kind == "cisd"
    assert staged.trial.data["ci1"].shape == mycc_frozen_list.t1.shape
    assert int(staged.trial.data["nocc_t_core"].item()) == 0
    assert int(staged.trial.data["nvir_t_outer"].item()) == 1


def test_stage_prefers_cc_mo_coeff_for_hamiltonian(mycc_rotated_basis):
    staged = Afqmc(mycc_rotated_basis).stage()

    c_cc = np.asarray(mycc_rotated_basis.mo_coeff)
    c_scf = np.asarray(mycc_rotated_basis._scf.mo_coeff)
    hcore = np.asarray(mycc_rotated_basis._scf.get_hcore())

    h1_cc = c_cc.T.conj() @ hcore @ c_cc
    h1_scf = c_scf.T.conj() @ hcore @ c_scf

    assert np.allclose(staged.ham.h1, h1_cc)
    assert not np.allclose(h1_cc, h1_scf)


@pytest.fixture(scope="module")
def mycc():
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
    mycc = cc.CCSD(mf)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_rotated_basis():
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

    nocc = mol.nelectron // 2
    nvir = mf.mo_coeff.shape[-1] - nocc

    rot_occ = np.eye(nocc)
    theta_occ = 0.31
    rot_occ[:2, :2] = np.array(
        [
            [np.cos(theta_occ), -np.sin(theta_occ)],
            [np.sin(theta_occ), np.cos(theta_occ)],
        ]
    )

    rot_vir = np.eye(nvir)
    theta_vir = -0.47
    rot_vir[:2, :2] = np.array(
        [
            [np.cos(theta_vir), -np.sin(theta_vir)],
            [np.sin(theta_vir), np.cos(theta_vir)],
        ]
    )

    mo_coeff = np.array(mf.mo_coeff, copy=True)
    mo_coeff[:, :nocc] = mo_coeff[:, :nocc] @ rot_occ
    mo_coeff[:, nocc:] = mo_coeff[:, nocc:] @ rot_vir

    mycc = cc.CCSD(mf, mo_coeff=mo_coeff)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_frozen_list():
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
    frozen = [0, int(mf.mo_coeff.shape[-1] - 1)]
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def mycc_frozen_ndarray():
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
    frozen = np.array([0, int(mf.mo_coeff.shape[-1] - 1)], dtype=np.int64)
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.kernel()
    return mycc


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
