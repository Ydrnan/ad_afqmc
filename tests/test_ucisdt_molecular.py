from trot import config

config.configure_once()

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
from trot.meas.ucisdt import make_ucisdt_meas_ops
from trot.staging import stage_from_ccpy
from trot.trial.ucisdt import make_ucisdt_trial_data, make_ucisdt_trial_ops


def _run_ucisdt_molecular_consistency(mol, *, spin_broken_uhf: bool = False):
    mf, driver = build_ccpy_driver(
        mol,
        cc_method="ccsdt",
        spin_broken_uhf=spin_broken_uhf,
    )
    staged = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = ham_from_staged(staged)

    trial_data = make_ucisdt_trial_data(staged.trial.data, sys)
    trial_ops = make_ucisdt_trial_ops(sys)

    meas_manual = make_ucisdt_meas_ops(sys, mixed_precision=False, testing=True)
    meas_auto = make_auto_meas_ops(sys, trial_ops, eps=1.0e-4)
    ctx_manual = meas_manual.build_meas_ctx(ham, trial_data)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial_data)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)
    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for walker in reference_and_rotated_walkers(trial_data):
        ovlp = trial_ops.overlap(walker, trial_data)
        assert jnp.isfinite(ovlp)

        v_m = fb_manual(walker, ham, ctx_manual, trial_data)
        v_a = fb_auto(walker, ham, ctx_auto, trial_data)
        assert jnp.allclose(v_a, v_m, rtol=5e-5, atol=5e-6), (v_a, v_m)

        e_m = e_manual(walker, ham, ctx_manual, trial_data)
        e_a = e_auto(walker, ham, ctx_auto, trial_data)
        assert jnp.allclose(e_a, e_m, rtol=5e-5, atol=5e-6), (e_a, e_m)


def _run_ucisdt_oracle_consistency(mol, *, spin_broken_uhf: bool = False):
    mf, driver = build_ccpy_driver(
        mol,
        cc_method="ccsdt",
        spin_broken_uhf=spin_broken_uhf,
    )
    staged = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = ham_from_staged(staged)
    trial_data = make_ucisdt_trial_data(staged.trial.data, sys)
    trial_ops = make_ucisdt_trial_ops(sys)
    meas = make_ucisdt_meas_ops(sys, mixed_precision=False, testing=True)
    e_manual = meas.require_kernel(k_energy)
    ctx = meas.build_meas_ctx(ham, trial_data)

    ci_amps = trial_to_oracle_ci_amps(trial_data, order=3)
    for walker in reference_and_rotated_walkers(trial_data):
        det_mo = walker_to_oracle_det_mo(mf, trial_data, walker)
        ov_o, e_o = oracle_overlap_energy(mf, ci_amps, det_mo)
        ov_t = trial_ops.overlap(walker, trial_data)
        e_t = e_manual(walker, ham, ctx, trial_data)

        assert jnp.allclose(ov_t, ov_o, atol=1e-10), (ov_t, ov_o)
        assert jnp.allclose(e_t, e_o, atol=1e-10), (e_t, e_o)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_kwargs,spin_broken_uhf",
    [
        (
            dict(atom="H 0 0 0; F 1.2 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0),
            False,
        ),
        (
            dict(atom="H 0 0 0; F 1.7 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0),
            True,
        ),
        (
            dict(
                atom=(
                    "O 0.0 0.0 0.0; " "H 0.0 0.0 2.2; " "H 0.95 0.13 -0.25; " "H -0.24 0.91 0.31"
                ),
                basis="3-21g",
                symmetry="c1",
                verbose=0,
                charge=1,
                spin=0,
            ),
            True,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; O 0 1.7 0",
                basis="3-21g",
                symmetry="c1",
                verbose=0,
                charge=1,
                spin=3,
            ),
            False,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; Be 0 1.7 0; He 7.0 0.0 0.0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
            ),
            False,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; H 0 1.7 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=1,
            ),
            False,
        ),
    ],
)
def test_ucisdt_molecular_ccsdt_consistency(mol_kwargs, spin_broken_uhf):
    mol = gto.M(**mol_kwargs)
    _run_ucisdt_molecular_consistency(mol, spin_broken_uhf=spin_broken_uhf)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_kwargs,spin_broken_uhf",
    [
        (
            dict(atom="H 0 0 0; F 1.2 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0),
            False,
        ),
        (
            dict(atom="H 0 0 0; F 1.7 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0),
            True,
        ),
        (
            dict(
                atom=(
                    "O 0.0 0.0 0.0; " "H 0.0 0.0 2.2; " "H 0.95 0.13 -0.25; " "H -0.24 0.91 0.31"
                ),
                basis="3-21g",
                symmetry="c1",
                verbose=0,
                charge=1,
                spin=0,
            ),
            True,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; O 0 1.7 0",
                basis="3-21g",
                symmetry="c1",
                verbose=0,
                charge=1,
                spin=3,
            ),
            False,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; Be 0 1.7 0; He 7.0 0.0 0.0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
            ),
            False,
        ),
        (
            dict(
                atom="H 0 0 0; H 0 0 1.1; H 0 1.7 0",
                basis="6-31g",
                symmetry="c1",
                verbose=0,
                charge=0,
                spin=1,
            ),
            False,
        ),
    ],
)
def test_ucisdt_molecular_matches_legacy_oracle(mol_kwargs, spin_broken_uhf):
    mol = gto.M(**mol_kwargs)
    _run_ucisdt_oracle_consistency(mol, spin_broken_uhf=spin_broken_uhf)
