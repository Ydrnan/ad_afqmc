import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import scf

from tests.helpers import hamiltonian_noci_overlaps, noci_overlaps
from trot.ham.chol import HamChol


def _run_uhf(mol, *, dm0=None):
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1.0e-14
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("UHF did not converge for molecular UCISDT/UCISDTQ test.")
    return mf


def _stability_rerun_uhf(mf):
    mo_i, _, _, _ = mf.stability(return_status=True)
    dm0 = mf.make_rdm1(mo_i, mf.mo_occ)
    mf = _run_uhf(mf.mol, dm0=dm0)

    _, _, stable_i, _ = mf.stability(return_status=True)
    if not stable_i:
        raise RuntimeError("UHF stability rerun did not converge to an internally stable solution.")

    s2 = mf.spin_square()[0]
    if abs(s2) < 1.0e-4:
        raise RuntimeError("UHF stability rerun did not converge to a spin-broken solution.")
    return mf


def build_ccpy_driver(mol, *, cc_method: str, spin_broken_uhf: bool = False):
    try:
        from ccpy.drivers.driver import Driver
    except Exception as exc:  # pragma: no cover - skip path depends on local ccpy build
        pytest.skip(f"ccpy is not available in this environment: {exc}")

    mf = _run_uhf(mol)
    if spin_broken_uhf:
        mf = _stability_rerun_uhf(mf)

    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.options["amp_convergence"] = 1.0e-14
    driver.options["energy_convergence"] = 1.0e-14
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method=cc_method)
    return mf, driver


def ham_from_staged(staged):
    return HamChol(
        h0=jnp.asarray(staged.ham.h0),
        h1=jnp.asarray(staged.ham.h1),
        chol=jnp.asarray(staged.ham.chol),
        basis=staged.ham.basis,
    )


def reference_and_rotated_walkers(trial_data, seed: int = 82):
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


def trial_to_oracle_ci_amps(trial_data, *, order: int):
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


def walker_to_oracle_det_mo(mf, trial_data, walker):
    wa, wb = walker
    c_a_ao = np.asarray(mf.mo_coeff[0])
    c_b_ao = np.asarray(mf.mo_coeff[1])
    c_b = np.asarray(trial_data.mo_coeff_b)

    wa_ao = c_a_ao @ np.asarray(wa)
    wb_b_basis = c_b.T @ np.asarray(wb)
    wb_ao = c_b_ao @ wb_b_basis
    return (wa_ao, wb_ao)


def oracle_overlap_energy(mf, ci_amps, det_mo):
    overlap = noci_overlaps.evaluate(mf, ci_amps, det_mo)
    h_overlap = hamiltonian_noci_overlaps.evaluate(mf, ci_amps, det_mo)
    energy = h_overlap / overlap + mf.mol.energy_nuc()
    return overlap, energy
