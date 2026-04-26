from trot import config

config.configure_once()

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, scf

from tests.helpers import hamiltonian_noci_overlaps, noci_overlaps
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.auto import make_auto_meas_ops
from trot.meas.ucisdt import make_ucisdt_meas_ops
from trot.meas.ucisdtq import make_ucisdtq_meas_ops
from trot.staging import stage_from_ccpy
from trot.trial.ucisdt import make_ucisdt_trial_data, make_ucisdt_trial_ops
from trot.trial.ucisdtq import make_ucisdtq_trial_data, make_ucisdtq_trial_ops


def _require_ccpy_driver():
    try:
        from ccpy.drivers.driver import Driver
    except Exception as exc:  # pragma: no cover - skip path depends on local ccpy build
        pytest.skip(f"ccpy is not available in this environment: {exc}")
    return Driver


def _build_ccpy_driver(mol, *, cc_method: str):
    Driver = _require_ccpy_driver()

    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.run(conv_tol=1.0e-12)
    if not mf.converged:
        raise RuntimeError("UHF did not converge for molecular UCISDT/UCISDTQ test.")

    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["amp_convergence"] = 1.0e-12
    driver.options["energy_convergence"] = 1.0e-12
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method=cc_method)
    return mf, driver


def _ham_from_staged(staged):
    return HamChol(
        h0=jnp.asarray(staged.ham.h0),
        h1=jnp.asarray(staged.ham.h1),
        chol=jnp.asarray(staged.ham.chol),
        basis=staged.ham.basis,
    )


def _reference_and_rotated_walkers(trial_data, seed: int = 82):
    nup, ndn = trial_data.nocc
    norb = trial_data.norb

    wa_ref = trial_data.mo_coeff_a[:, :nup]
    wb_ref = trial_data.mo_coeff_b[:, :ndn]

    rng = np.random.default_rng(seed)
    rand_a = jnp.asarray(rng.standard_normal((norb, norb)))
    rand_b = jnp.asarray(rng.standard_normal((norb, norb)))

    wa_rot = (trial_data.mo_coeff_a @ rand_a)[:, :nup]
    wb_rot = (trial_data.mo_coeff_b @ rand_b)[:, :ndn]

    return [(wa_ref, wb_ref), (wa_rot, wb_rot)]


def _trial_to_oracle_ci_amps(trial_data, *, order: int):
    ci_amps = [
        [np.float64(1.0)],
        [np.asarray(trial_data.c1a), np.asarray(trial_data.c1b)],
        [
            np.asarray(trial_data.c2aa).transpose(0, 2, 1, 3),
            np.asarray(trial_data.c2ab).transpose(0, 2, 1, 3),
            np.asarray(trial_data.c2bb).transpose(0, 2, 1, 3),
        ],
        [
            np.asarray(trial_data.c3aaa).transpose(0, 2, 4, 1, 3, 5),
            np.asarray(trial_data.c3aab).transpose(0, 2, 4, 1, 3, 5),
            np.asarray(trial_data.c3abb).transpose(0, 2, 4, 1, 3, 5),
            np.asarray(trial_data.c3bbb).transpose(0, 2, 4, 1, 3, 5),
        ],
    ]
    if order >= 4:
        ci_amps.append(
            [
                np.asarray(trial_data.c4aaaa).transpose(0, 2, 4, 6, 1, 3, 5, 7),
                np.asarray(trial_data.c4aaab).transpose(0, 2, 4, 6, 1, 3, 5, 7),
                np.asarray(trial_data.c4aabb).transpose(0, 2, 4, 6, 1, 3, 5, 7),
                np.asarray(trial_data.c4abbb).transpose(0, 2, 4, 6, 1, 3, 5, 7),
                np.asarray(trial_data.c4bbbb).transpose(0, 2, 4, 6, 1, 3, 5, 7),
            ]
        )
    return ci_amps


def _walker_to_oracle_det_mo(mf, trial_data, walker):
    wa, wb = walker
    c_a_ao = np.asarray(mf.mo_coeff[0])
    c_b_ao = np.asarray(mf.mo_coeff[1])
    c_b = np.asarray(trial_data.mo_coeff_b)

    wa_ao = c_a_ao @ np.asarray(wa)
    wb_b_basis = c_b.T @ np.asarray(wb)
    wb_ao = c_b_ao @ wb_b_basis
    return (wa_ao, wb_ao)


def _oracle_overlap_energy(mf, ci_amps, det_mo):
    overlap = noci_overlaps.evaluate(mf, ci_amps, det_mo)
    h_overlap = hamiltonian_noci_overlaps.evaluate(mf, ci_amps, det_mo)
    energy = h_overlap / overlap + mf.mol.energy_nuc()
    return overlap, energy


def _run_ucisdt_molecular_consistency(mol):
    mf, driver = _build_ccpy_driver(mol, cc_method="ccsdt")
    staged = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = _ham_from_staged(staged)

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

    for walker in _reference_and_rotated_walkers(trial_data):
        ovlp = trial_ops.overlap(walker, trial_data)
        assert jnp.isfinite(ovlp)

        v_m = fb_manual(walker, ham, ctx_manual, trial_data)
        v_a = fb_auto(walker, ham, ctx_auto, trial_data)
        assert jnp.allclose(v_a, v_m, rtol=5e-5, atol=5e-6), (v_a, v_m)

        e_m = e_manual(walker, ham, ctx_manual, trial_data)
        e_a = e_auto(walker, ham, ctx_auto, trial_data)
        assert jnp.allclose(e_a, e_m, rtol=5e-5, atol=5e-6), (e_a, e_m)


def _run_ucisdtq_molecular_consistency(mol):
    mf, driver = _build_ccpy_driver(mol, cc_method="ccsdt")
    staged_q = stage_from_ccpy(driver, mf, order=4, chol_cut=1.0e-14, verbose=False)
    staged_t = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged_q.ham.norb), nelec=staged_q.ham.nelec, walker_kind="unrestricted")
    ham = _ham_from_staged(staged_q)

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

    for walker in _reference_and_rotated_walkers(trial_q0):
        ovlp_t = trial_t_ops.overlap(walker, trial_t)
        ovlp_q = trial_q_ops.overlap(walker, trial_q0)
        assert jnp.allclose(ovlp_q, ovlp_t, rtol=5e-6, atol=5e-7), (ovlp_q, ovlp_t)

        v_t = fb_t(walker, ham, ctx_t, trial_t)
        v_q = fb_q(walker, ham, ctx_q, trial_q0)
        assert jnp.allclose(v_q, v_t, rtol=5e-5, atol=5e-6), (v_q, v_t)

        e_t_val = e_t(walker, ham, ctx_t, trial_t)
        e_q_val = e_q(walker, ham, ctx_q, trial_q0)
        assert jnp.allclose(e_q_val, e_t_val, rtol=5e-5, atol=5e-6), (e_q_val, e_t_val)


def _run_ucisdt_oracle_consistency(mol):
    mf, driver = _build_ccpy_driver(mol, cc_method="ccsdt")
    staged = stage_from_ccpy(driver, mf, order=3, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = _ham_from_staged(staged)
    trial_data = make_ucisdt_trial_data(staged.trial.data, sys)
    trial_ops = make_ucisdt_trial_ops(sys)
    e_manual = make_ucisdt_meas_ops(sys, mixed_precision=False, testing=True).require_kernel(
        k_energy
    )
    ctx = make_ucisdt_meas_ops(sys, mixed_precision=False, testing=True).build_meas_ctx(
        ham, trial_data
    )

    ci_amps = _trial_to_oracle_ci_amps(trial_data, order=3)
    for walker in _reference_and_rotated_walkers(trial_data):
        det_mo = _walker_to_oracle_det_mo(mf, trial_data, walker)
        ov_o, e_o = _oracle_overlap_energy(mf, ci_amps, det_mo)
        ov_t = trial_ops.overlap(walker, trial_data)
        e_t = e_manual(walker, ham, ctx, trial_data)

        assert jnp.allclose(ov_t, ov_o, atol=1e-10), (ov_t, ov_o)
        assert jnp.allclose(e_t, e_o, atol=1e-10), (e_t, e_o)


def _run_ucisdtq_oracle_consistency(mol):
    mf, driver = _build_ccpy_driver(mol, cc_method="ccsdt")
    staged = stage_from_ccpy(driver, mf, order=4, chol_cut=1.0e-14, verbose=False)

    sys = System(norb=int(staged.ham.norb), nelec=staged.ham.nelec, walker_kind="unrestricted")
    ham = _ham_from_staged(staged)
    trial_data = make_ucisdtq_trial_data(staged.trial.data, sys)
    trial_ops = make_ucisdtq_trial_ops(sys)
    meas = make_ucisdtq_meas_ops(sys, trial_ops)
    ctx = meas.build_meas_ctx(ham, trial_data)
    e_kernel = meas.require_kernel(k_energy)

    ci_amps = _trial_to_oracle_ci_amps(trial_data, order=4)
    for walker in _reference_and_rotated_walkers(trial_data):
        det_mo = _walker_to_oracle_det_mo(mf, trial_data, walker)
        ov_o, e_o = _oracle_overlap_energy(mf, ci_amps, det_mo)
        ov_t = trial_ops.overlap(walker, trial_data)
        e_t = e_kernel(walker, ham, ctx, trial_data)

        assert jnp.allclose(ov_t, ov_o, atol=1e-10), (ov_t, ov_o)
        # UCISDTQ measurement uses finite differences, so keep relaxed tolerance.
        assert jnp.allclose(e_t, e_o, rtol=5e-5, atol=1e-5), (e_t, e_o)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_kwargs",
    [
        dict(atom="H 0 0 0; F 1.2 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0),
        dict(
            atom="H 0 0 0; H 0 0 1.1; O 0 1.7 0",
            basis="3-21g",
            symmetry="c1",
            verbose=0,
            charge=1,
            spin=3,
        ),
    ],
)
def test_ucisdt_molecular_ccsdt_consistency(mol_kwargs):
    mol = gto.M(**mol_kwargs)
    _run_ucisdt_molecular_consistency(mol)


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
def test_ucisdt_molecular_matches_legacy_oracle():
    mol = gto.M(atom="H 0 0 0; F 1.2 0.0 0.0", basis="6-31g", symmetry="c1", verbose=0)
    _run_ucisdt_oracle_consistency(mol)


@pytest.mark.integration
@pytest.mark.slow
def test_ucisdtq_molecular_matches_legacy_oracle():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.1; H 0 1.7 0",
        basis="6-31g",
        symmetry="c1",
        verbose=0,
        charge=0,
        spin=1,
    )
    _run_ucisdtq_oracle_consistency(mol)
