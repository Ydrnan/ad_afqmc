"""
Unit tests for ccpy T-amplitude → CI-amplitude conversion.

These tests use a mock ccpy driver (no ccpy installation required) and
check shapes, symmetry properties, and known limiting cases of the
_ccpy_t_to_c_amplitudes helper in trot.staging.
"""

import numpy as np
import pytest

from trot.staging import _ccpy_t_to_c_amplitudes, stage_from_ccpy


# ---------------------------------------------------------------------------
# Mock ccpy driver
# ---------------------------------------------------------------------------


class _MockT:
    pass


def _make_driver(n_oa, n_va, n_ob, n_vb, *, order_cc: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    d = type("Driver", (), {"operator_params": {"order": order_cc}, "T": _MockT()})()

    d.T.a = rng.standard_normal((n_va, n_oa))
    d.T.b = rng.standard_normal((n_vb, n_ob))
    d.T.aa = rng.standard_normal((n_va, n_va, n_oa, n_oa))
    d.T.bb = rng.standard_normal((n_vb, n_vb, n_ob, n_ob))
    d.T.ab = rng.standard_normal((n_va, n_vb, n_oa, n_ob))

    if order_cc >= 3:
        d.T.aaa = rng.standard_normal((n_va, n_va, n_va, n_oa, n_oa, n_oa))
        d.T.aab = rng.standard_normal((n_va, n_va, n_vb, n_oa, n_oa, n_ob))
        d.T.abb = rng.standard_normal((n_va, n_vb, n_vb, n_oa, n_ob, n_ob))
        d.T.bbb = rng.standard_normal((n_vb, n_vb, n_vb, n_ob, n_ob, n_ob))

    if order_cc >= 4:
        d.T.aaaa = rng.standard_normal((n_va, n_va, n_va, n_va, n_oa, n_oa, n_oa, n_oa))
        d.T.aaab = rng.standard_normal((n_va, n_va, n_va, n_vb, n_oa, n_oa, n_oa, n_ob))
        d.T.aabb = rng.standard_normal((n_va, n_va, n_vb, n_vb, n_oa, n_oa, n_ob, n_ob))
        d.T.abbb = rng.standard_normal((n_va, n_vb, n_vb, n_vb, n_oa, n_ob, n_ob, n_ob))
        d.T.bbbb = rng.standard_normal((n_vb, n_vb, n_vb, n_vb, n_ob, n_ob, n_ob, n_ob))

    return d


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_oa,n_va,n_ob,n_vb", [(2, 3, 2, 3), (3, 2, 2, 2)])
def test_amplitude_shapes_order3(n_oa, n_va, n_ob, n_vb):
    driver = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=3)
    amps = _ccpy_t_to_c_amplitudes(driver, order=3, order_cc=3)

    assert set(amps.keys()) == {
        "ci1a",
        "ci1b",
        "ci2aa",
        "ci2ab",
        "ci2bb",
        "ci3aaa",
        "ci3aab",
        "ci3abb",
        "ci3bbb",
    }
    assert amps["ci1a"].shape == (n_oa, n_va)
    assert amps["ci1b"].shape == (n_ob, n_vb)
    assert amps["ci2aa"].shape == (n_oa, n_va, n_oa, n_va)
    assert amps["ci2ab"].shape == (n_oa, n_va, n_ob, n_vb)
    assert amps["ci2bb"].shape == (n_ob, n_vb, n_ob, n_vb)
    assert amps["ci3aaa"].shape == (n_oa, n_va, n_oa, n_va, n_oa, n_va)
    assert amps["ci3aab"].shape == (n_oa, n_va, n_oa, n_va, n_ob, n_vb)
    assert amps["ci3abb"].shape == (n_oa, n_va, n_ob, n_vb, n_ob, n_vb)
    assert amps["ci3bbb"].shape == (n_ob, n_vb, n_ob, n_vb, n_ob, n_vb)


@pytest.mark.parametrize("n_oa,n_va,n_ob,n_vb", [(2, 3, 2, 3)])
def test_amplitude_shapes_order4(n_oa, n_va, n_ob, n_vb):
    driver = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=4)
    amps = _ccpy_t_to_c_amplitudes(driver, order=4, order_cc=4)

    assert set(amps.keys()) == {
        "ci1a",
        "ci1b",
        "ci2aa",
        "ci2ab",
        "ci2bb",
        "ci3aaa",
        "ci3aab",
        "ci3abb",
        "ci3bbb",
        "ci4aaaa",
        "ci4aaab",
        "ci4aabb",
        "ci4abbb",
        "ci4bbbb",
    }
    assert amps["ci4aaaa"].shape == (n_oa, n_va, n_oa, n_va, n_oa, n_va, n_oa, n_va)
    assert amps["ci4aaab"].shape == (n_oa, n_va, n_oa, n_va, n_oa, n_va, n_ob, n_vb)
    assert amps["ci4aabb"].shape == (n_oa, n_va, n_oa, n_va, n_ob, n_vb, n_ob, n_vb)
    assert amps["ci4abbb"].shape == (n_oa, n_va, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb)
    assert amps["ci4bbbb"].shape == (n_ob, n_vb, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb)


# ---------------------------------------------------------------------------
# Antisymmetry tests
# ---------------------------------------------------------------------------


def test_c2aa_antisymmetry():
    """C2aa must be antisymmetric under swap of the two virtual indices."""
    driver = _make_driver(2, 3, 2, 3, order_cc=2)
    amps = _ccpy_t_to_c_amplitudes(driver, order=2, order_cc=2)
    C2aa = amps["ci2aa"]
    # (iajb) -> (ibjа) swap: transpose(0,3,2,1) should negate
    np.testing.assert_allclose(C2aa, -C2aa.transpose(0, 3, 2, 1), atol=1e-12)


def test_c2bb_antisymmetry():
    driver = _make_driver(2, 3, 2, 3, order_cc=2)
    amps = _ccpy_t_to_c_amplitudes(driver, order=2, order_cc=2)
    C2bb = amps["ci2bb"]
    np.testing.assert_allclose(C2bb, -C2bb.transpose(0, 3, 2, 1), atol=1e-12)


def test_c4aaaa_antisymmetry_in_virtuals():
    """C4aaaa must be antisymmetric under any swap of two virtual (a) indices."""
    driver = _make_driver(2, 3, 2, 3, order_cc=4)
    amps = _ccpy_t_to_c_amplitudes(driver, order=4, order_cc=4)
    C4 = amps["ci4aaaa"]
    # Swap a <-> b (indices 1 and 3)
    np.testing.assert_allclose(C4, -C4.transpose(0, 3, 2, 1, 4, 5, 6, 7), atol=1e-10)
    # Swap a <-> c (indices 1 and 5)
    np.testing.assert_allclose(C4, -C4.transpose(0, 5, 2, 3, 4, 1, 6, 7), atol=1e-10)


def test_c4bbbb_antisymmetry_in_virtuals():
    driver = _make_driver(2, 3, 2, 3, order_cc=4)
    amps = _ccpy_t_to_c_amplitudes(driver, order=4, order_cc=4)
    C4 = amps["ci4bbbb"]
    np.testing.assert_allclose(C4, -C4.transpose(0, 3, 2, 1, 4, 5, 6, 7), atol=1e-10)


# ---------------------------------------------------------------------------
# Zero-filling when order_cc < requested order
# ---------------------------------------------------------------------------


def test_t3_zeroed_when_order_cc_lt_3():
    """When order_cc=2, the connected T3 contribution is zero and C3 = disconnected only."""
    n_oa, n_va, n_ob, n_vb = 2, 2, 2, 2
    driver_cc2 = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=2, rng=np.random.default_rng(42))
    driver_cc3 = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=3, rng=np.random.default_rng(42))
    # zero out T3 on the cc3 driver to make them equivalent
    driver_cc3.T.aaa[:] = 0.0
    driver_cc3.T.aab[:] = 0.0
    driver_cc3.T.abb[:] = 0.0
    driver_cc3.T.bbb[:] = 0.0

    amps_cc2 = _ccpy_t_to_c_amplitudes(driver_cc2, order=3, order_cc=2)
    amps_cc3 = _ccpy_t_to_c_amplitudes(driver_cc3, order=3, order_cc=3)

    for key in ("ci3aaa", "ci3aab", "ci3abb", "ci3bbb"):
        np.testing.assert_allclose(
            amps_cc2[key], amps_cc3[key], atol=1e-12, err_msg=f"Mismatch in {key}"
        )


def test_t4_zeroed_when_order_cc_lt_4():
    """When order_cc=3, the connected T4 is zero; C4 equals the disconnected part only."""
    n_oa, n_va, n_ob, n_vb = 2, 2, 2, 2
    rng = np.random.default_rng(99)
    driver_cc3 = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=3, rng=rng)
    driver_cc4 = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=4, rng=np.random.default_rng(99))
    # Zero T4 on the cc4 driver
    for attr in ("aaaa", "aaab", "aabb", "abbb", "bbbb"):
        getattr(driver_cc4.T, attr)[:] = 0.0

    amps_cc3 = _ccpy_t_to_c_amplitudes(driver_cc3, order=4, order_cc=3)
    amps_cc4 = _ccpy_t_to_c_amplitudes(driver_cc4, order=4, order_cc=4)

    for key in ("ci4aaaa", "ci4aaab", "ci4aabb", "ci4abbb", "ci4bbbb"):
        np.testing.assert_allclose(
            amps_cc3[key], amps_cc4[key], atol=1e-12, err_msg=f"Mismatch in {key}"
        )


# ---------------------------------------------------------------------------
# Known limiting case: only T1 non-zero
# ---------------------------------------------------------------------------


def test_c2_when_only_t1():
    """When T2 = 0, C2aa = antisym(2 * t1 ⊗ t1), C2ab = t1a ⊗ t1b."""
    n_oa, n_va, n_ob, n_vb = 2, 3, 2, 3
    rng = np.random.default_rng(7)

    driver = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=2, rng=rng)
    driver.T.aa[:] = 0.0
    driver.T.bb[:] = 0.0
    driver.T.ab[:] = 0.0

    t1a = driver.T.a.T  # (n_oa, n_va)
    t1b = driver.T.b.T

    amps = _ccpy_t_to_c_amplitudes(driver, order=2, order_cc=2)

    expected_c2aa_raw = 2.0 * np.einsum("ia,jb->iajb", t1a, t1a)
    expected_c2aa = 0.5 * (expected_c2aa_raw - expected_c2aa_raw.transpose(0, 3, 2, 1))
    np.testing.assert_allclose(amps["ci2aa"], expected_c2aa, atol=1e-12)

    expected_c2ab = np.einsum("ia,jb->iajb", t1a, t1b)
    np.testing.assert_allclose(amps["ci2ab"], expected_c2ab, atol=1e-12)


def test_order2_returns_no_triples_keys():
    """Requesting order=2 returns exactly the UCISD set of keys."""
    driver = _make_driver(2, 3, 2, 3, order_cc=2)
    amps = _ccpy_t_to_c_amplitudes(driver, order=2, order_cc=2)
    assert set(amps.keys()) == {"ci1a", "ci1b", "ci2aa", "ci2ab", "ci2bb"}


def test_stage_from_ccpy_order2_builds_ucisd_trial():
    """stage_from_ccpy(order=2) should build a UCISD trial from ccpy-like T amplitudes."""
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, symmetry="c1", verbose=0)
    mf = scf.UHF(mol)
    mf.run(conv_tol=1.0e-12)
    assert mf.converged

    n_oa, n_ob = map(int, mol.nelec)
    nmo = int(mf.mo_coeff[0].shape[1])
    n_va = nmo - n_oa
    n_vb = nmo - n_ob

    driver = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=2, rng=np.random.default_rng(17))
    staged = stage_from_ccpy(driver, mf, order=2, chol_cut=1.0e-8, verbose=False)

    assert staged.trial.kind == "ucisd"
    assert set(staged.trial.data.keys()) == {
        "mo_coeff_a",
        "mo_coeff_b",
        "ci1a",
        "ci1b",
        "ci2aa",
        "ci2ab",
        "ci2bb",
    }
    assert staged.trial.data["ci1a"].shape == (n_oa, n_va)
    assert staged.trial.data["ci1b"].shape == (n_ob, n_vb)
    assert staged.trial.data["ci2aa"].shape == (n_oa, n_va, n_oa, n_va)
    assert staged.trial.data["ci2ab"].shape == (n_oa, n_va, n_ob, n_vb)
    assert staged.trial.data["ci2bb"].shape == (n_ob, n_vb, n_ob, n_vb)


def test_stage_from_ccpy_order_lt_2_raises():
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, symmetry="c1", verbose=0)
    mf = scf.UHF(mol)
    mf.run(conv_tol=1.0e-12)
    assert mf.converged

    n_oa, n_ob = map(int, mol.nelec)
    nmo = int(mf.mo_coeff[0].shape[1])
    n_va = nmo - n_oa
    n_vb = nmo - n_ob
    driver = _make_driver(n_oa, n_va, n_ob, n_vb, order_cc=2, rng=np.random.default_rng(5))

    with pytest.raises(ValueError, match="requires order >= 2"):
        stage_from_ccpy(driver, mf, order=1, chol_cut=1.0e-8, verbose=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
