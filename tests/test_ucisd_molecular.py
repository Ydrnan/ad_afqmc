from trot import config

config.configure_once()

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.ucisd import make_ucisd_meas_ops
from trot.staging import stage, stage_from_ccpy
from trot.trial.ucisd import make_ucisd_trial_data, make_ucisd_trial_ops


def _require_ccpy_driver():
    try:
        from ccpy.drivers.driver import Driver
    except Exception as exc:  # pragma: no cover - skip path depends on local ccpy build
        pytest.skip(f"ccpy is not available in this environment: {exc}")
    return Driver


def _build_pyscf_and_ccpy_ccsd(mol):
    Driver = _require_ccpy_driver()

    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.run(conv_tol=1.0e-12)
    if not mf.converged:
        raise RuntimeError("UHF did not converge for UCISD molecular comparison test.")

    mycc = cc.UCCSD(mf)
    mycc.max_cycle = 300
    mycc.conv_tol = 1.0e-12
    mycc.conv_tol_normt = 1.0e-10
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("PySCF UCCSD did not converge for UCISD molecular comparison test.")

    ccpy_driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    ccpy_driver.options["amp_convergence"] = 1.0e-12
    ccpy_driver.options["energy_convergence"] = 1.0e-12
    ccpy_driver.options["RHF_symmetry"] = False
    ccpy_driver.run_cc(method="ccsd")

    return mf, mycc, ccpy_driver


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


def _v0_from_chol(chol):
    return 0.5 * jnp.einsum("gik,gjk->ij", chol, chol, optimize="optimal")


@pytest.mark.integration
@pytest.mark.slow
def test_ucisd_molecular_ccsd_pyscf_vs_ccpy():
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
        symmetry="c1",
        verbose=0,
    )
    mf, mycc, ccpy_driver = _build_pyscf_and_ccpy_ccsd(mol)

    staged_pyscf = stage(mycc, chol_cut=1.0e-14, verbose=False)
    staged_ccpy = stage_from_ccpy(ccpy_driver, mf, order=2, chol_cut=1.0e-14, verbose=False)

    assert staged_pyscf.trial.kind == "ucisd"
    assert staged_ccpy.trial.kind == "ucisd"
    assert staged_pyscf.ham.norb == staged_ccpy.ham.norb
    assert staged_pyscf.ham.nelec == staged_ccpy.ham.nelec

    ham_pyscf = _ham_from_staged(staged_pyscf)
    ham_ccpy = _ham_from_staged(staged_ccpy)

    # Cholesky vectors can differ by signs/orderings while representing the same operator.
    assert jnp.allclose(ham_ccpy.h0, ham_pyscf.h0, rtol=0.0, atol=1e-10)
    assert jnp.allclose(ham_ccpy.h1, ham_pyscf.h1, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(_v0_from_chol(ham_ccpy.chol), _v0_from_chol(ham_pyscf.chol), rtol=1e-6, atol=1e-8)

    sys = System(
        norb=int(staged_ccpy.ham.norb),
        nelec=staged_ccpy.ham.nelec,
        walker_kind="unrestricted",
    )
    trial_ccpy = make_ucisd_trial_data(staged_ccpy.trial.data, sys)
    trial_pyscf = make_ucisd_trial_data(staged_pyscf.trial.data, sys)
    trial_ops = make_ucisd_trial_ops(sys)

    meas_ops = make_ucisd_meas_ops(sys, mixed_precision=False, testing=True)
    ctx_ccpy = meas_ops.build_meas_ctx(ham_ccpy, trial_ccpy)
    ctx_pyscf = meas_ops.build_meas_ctx(ham_pyscf, trial_pyscf)
    fb_kernel = meas_ops.require_kernel(k_force_bias)
    e_kernel = meas_ops.require_kernel(k_energy)

    for walker in _reference_and_rotated_walkers(trial_ccpy):
        ovlp_ccpy = trial_ops.overlap(walker, trial_ccpy)
        ovlp_pyscf = trial_ops.overlap(walker, trial_pyscf)
        assert jnp.allclose(ovlp_ccpy, ovlp_pyscf, rtol=1e-5, atol=1e-7), (ovlp_ccpy, ovlp_pyscf)

        fb_ccpy = fb_kernel(walker, ham_ccpy, ctx_ccpy, trial_ccpy)
        fb_pyscf = fb_kernel(walker, ham_pyscf, ctx_pyscf, trial_pyscf)
        assert jnp.allclose(fb_ccpy, fb_pyscf, rtol=2e-4, atol=2e-6), (fb_ccpy, fb_pyscf)

        e_ccpy = e_kernel(walker, ham_ccpy, ctx_ccpy, trial_ccpy)
        e_pyscf = e_kernel(walker, ham_pyscf, ctx_pyscf, trial_pyscf)
        assert jnp.allclose(e_ccpy, e_pyscf, rtol=2e-4, atol=2e-6), (e_ccpy, e_pyscf)
