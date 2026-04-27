from trot import config

config.configure_once()

from dataclasses import replace

import jax.numpy as jnp
import pytest
from pyscf import gto

from tests.helpers.ucisdtx_molecular import (
    build_ccpy_driver,
    ham_from_staged,
    oracle_overlap_energy,
    reference_and_rotated_walkers,
    trial_to_oracle_ci_amps,
    walker_to_oracle_det_mo,
)
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.auto import make_auto_meas_ops
from trot.meas.ucisdtq import make_ucisdtq_meas_ops
from trot.staging import stage_from_ccpy
from trot.trial.ucisdt import make_ucisdt_trial_data, make_ucisdt_trial_ops
from trot.trial.ucisdtq import make_ucisdtq_trial_data, make_ucisdtq_trial_ops


def _run_ucisdtq_molecular_consistency(mol):
    mf, driver = build_ccpy_driver(mol, cc_method="ccsdt")
    staged_q = stage_from_ccpy(driver, mf, order=4, chol_cut=1.0e-14, verbose=False)
    staged_t = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged_q.ham.norb), nelec=staged_q.ham.nelec, walker_kind="unrestricted")
    ham = ham_from_staged(staged_q)

    trial_t = make_ucisdt_trial_data(staged_t.trial.data, sys)
    trial_t_ops = make_ucisdt_trial_ops(sys)

    trial_q = make_ucisdtq_trial_data(staged_q.trial.data, sys)
    trial_q_ops = make_ucisdtq_trial_ops(sys)

    # This is a reduction test for the UCISDTQ kernels, not a staging test.
    # A CCSDT input staged with order=4 contains disconnected C4 amplitudes;
    # here we intentionally remove all CI quadruples so the trial is UCISDT.
    trial_q0 = replace(
        trial_q,
        c4aaaa=jnp.zeros_like(trial_q.c4aaaa),
        c4aaab=jnp.zeros_like(trial_q.c4aaab),
        c4aabb=jnp.zeros_like(trial_q.c4aabb),
        c4abbb=jnp.zeros_like(trial_q.c4abbb),
        c4bbbb=jnp.zeros_like(trial_q.c4bbbb),
    )

    meas_t = make_auto_meas_ops(sys, trial_t_ops, eps=1.0e-4)
    meas_q = make_ucisdtq_meas_ops(sys, trial_q_ops)
    ctx_t = meas_t.build_meas_ctx(ham, trial_t)
    ctx_q = meas_q.build_meas_ctx(ham, trial_q0)
    fb_t = meas_t.require_kernel(k_force_bias)
    fb_q = meas_q.require_kernel(k_force_bias)
    e_t = meas_t.require_kernel(k_energy)
    e_q = meas_q.require_kernel(k_energy)

    for walker in reference_and_rotated_walkers(trial_q0):
        ovlp_t = trial_t_ops.overlap(walker, trial_t)
        ovlp_q = trial_q_ops.overlap(walker, trial_q0)
        assert jnp.allclose(ovlp_q, ovlp_t, rtol=5e-6, atol=5e-7), (ovlp_q, ovlp_t)

        v_t = fb_t(walker, ham, ctx_t, trial_t)
        v_q = fb_q(walker, ham, ctx_q, trial_q0)
        assert jnp.allclose(v_q, v_t, rtol=5e-5, atol=5e-6), (v_q, v_t)

        e_t_val = e_t(walker, ham, ctx_t, trial_t)
        e_q_val = e_q(walker, ham, ctx_q, trial_q0)
        assert jnp.allclose(e_q_val, e_t_val, rtol=5e-5, atol=5e-6), (e_q_val, e_t_val)


def _run_ucisdtq_oracle_consistency(
    mol,
    *,
    cc_method: str = "ccsdt",
    order: int = 4,
    rotated_eq_unrotated: bool = False,
):
    mf, driver = build_ccpy_driver(mol, cc_method=cc_method)
    staged = stage_from_ccpy(driver, mf, order=order, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = ham_from_staged(staged)
    trial_data = make_ucisdtq_trial_data(staged.trial.data, sys)
    trial_ops = make_ucisdtq_trial_ops(sys)
    meas = make_ucisdtq_meas_ops(sys, trial_ops)
    ctx = meas.build_meas_ctx(ham, trial_data)
    e_kernel = meas.require_kernel(k_energy)

    ci_amps = trial_to_oracle_ci_amps(trial_data, order=4)
    ref_energy = None
    for iw, walker in enumerate(reference_and_rotated_walkers(trial_data)):
        det_mo = walker_to_oracle_det_mo(mf, trial_data, walker)
        ov_o, e_o = oracle_overlap_energy(mf, ci_amps, det_mo)
        ov_t = trial_ops.overlap(walker, trial_data)
        e_t = e_kernel(walker, ham, ctx, trial_data)

        assert jnp.allclose(ov_t, ov_o, atol=1e-10), (ov_t, ov_o)
        # UCISDTQ measurement uses finite differences, so keep relaxed tolerance.
        assert jnp.allclose(e_t, e_o, rtol=5e-5, atol=1e-5), (e_t, e_o)
        if iw == 0:
            ref_energy = e_t
        elif rotated_eq_unrotated:
            assert ref_energy is not None
            assert jnp.allclose(e_t, ref_energy, rtol=5e-5, atol=1e-5), (e_t, ref_energy)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_kwargs",
    [
        dict(
            atom="H 0 0 0; H 0 0 1.1; H 0 1.7 0",
            basis="6-31g",
            symmetry="c1",
            verbose=0,
            charge=0,
            spin=1,
        ),
    ],
)
def test_ucisdtq_molecular_reduces_to_ucisdt_when_q_zero(mol_kwargs):
    mol = gto.M(**mol_kwargs)
    _run_ucisdtq_molecular_consistency(mol)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_kwargs,cc_method,order,rotated_eq_unrotated",
    [
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; He 0 1.7 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsdtq",
            4,
            True,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; He 0 1.7 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=2,
            ),
            "ccsdtq",
            4,
            True,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; Be 0 1.7 0",
                basis="sto-3g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsdtq",
            4,
            False,
        ),
        (
            dict(
                atom="He 0 0 0; He 0 0 1.1; He 0 1.7 0; He 1.1 0 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsdtq",
            4,
            False,
        ),
        (
            dict(
                atom="He 0 0 0; He 0 0 1.1; He 0 1.7 0; He 1.1 0 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsdt",
            4,
            False,
        ),
        (
            dict(
                atom="He 0 0 0; He 0 0 1.1; He 0 1.7 0; He 1.1 0 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsd",
            4,
            False,
        ),
        (
            dict(
                atom=(
                    "H 0.00  0.00  0.00; "
                    "H 1.15  0.07  0.02; "
                    "H 2.31 -0.05  0.11; "
                    "H 3.48  0.09 -0.06; "
                    "H 4.66 -0.08  0.17; "
                    "H 5.85  0.12 -0.10; "
                    "H 7.03 -0.04  0.06; "
                    "H 8.24  0.06 -0.15"
                ),
                basis="sto-3g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=0,
            ),
            "ccsdtq",
            4,
            False,
        ),
    ],
)
def test_ucisdtq_molecular_matches_legacy_oracle(
    mol_kwargs, cc_method, order, rotated_eq_unrotated
):
    mol = gto.M(**mol_kwargs)
    _run_ucisdtq_oracle_consistency(
        mol,
        cc_method=cc_method,
        order=order,
        rotated_eq_unrotated=rotated_eq_unrotated,
    )
