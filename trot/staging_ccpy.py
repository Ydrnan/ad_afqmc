from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from numpy.typing import NDArray

from .staging import (
    STAGE_FORMAT_VERSION,
    StagedInputs,
    StagedMfOrCc,
    TrialInput,
    _freeze_meta_value,
    _mf_coeff_helper,
    _resolve_stage_frozen_arg,
    _stage_begin,
    _stage_end,
    _stage_ham_input,
    _stage_ham_input_from_fcidump,
    dump,
    load,
)


def _ccpy_perm_sign(perm) -> int:
    sign = 1
    p = list(perm)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                sign *= -1
    return sign


def _ccpy_antisym_2(C2: NDArray) -> NDArray:
    return 0.5 * (C2 - C2.transpose(0, 3, 2, 1))


def _ccpy_sym_pairs_3(C3: NDArray) -> NDArray:
    return (1 / 6) * (
        C3
        + C3.transpose(0, 1, 4, 5, 2, 3)
        + C3.transpose(2, 3, 4, 5, 0, 1)
        + C3.transpose(2, 3, 0, 1, 4, 5)
        + C3.transpose(4, 5, 0, 1, 2, 3)
        + C3.transpose(4, 5, 2, 3, 0, 1)
    )


def _ccpy_antisym_3(C3: NDArray) -> NDArray:
    return (1 / 6) * (
        C3
        - C3.transpose(0, 1, 2, 5, 4, 3)
        + C3.transpose(0, 3, 2, 5, 4, 1)
        - C3.transpose(0, 3, 2, 1, 4, 5)
        + C3.transpose(0, 5, 2, 1, 4, 3)
        - C3.transpose(0, 5, 2, 3, 4, 1)
    )


def _ccpy_sym_pairs_4(C4: NDArray) -> NDArray:
    result = np.zeros_like(C4)
    for p in itertools.permutations([0, 1, 2, 3]):
        axes = []
        for pair in p:
            axes.extend([2 * pair, 2 * pair + 1])
        result = result + C4.transpose(*axes)
    return result / 24


def _ccpy_antisym_4(C4: NDArray) -> NDArray:
    antisym_axes = [1, 3, 5, 7]
    result = np.zeros_like(C4)
    for perm in itertools.permutations(antisym_axes):
        new_order = [0, perm[0], 2, perm[1], 4, perm[2], 6, perm[3]]
        sign = _ccpy_perm_sign(perm)
        result = result + sign * C4.transpose(new_order)
    return result / 24


def _ccpy_t_to_c_amplitudes(driver: Any, order: int, order_cc: int) -> dict:
    """
    Convert ccpy T amplitudes to CI amplitude dict suitable for UCISDT/UCISDTQ TrialInput.

    order: desired CI excitation level (3 or 4)
    order_cc: the CC level used in the ccpy calculation (may be lower than order)
    """
    # T1: ccpy stores (vir, occ); transpose to (occ, vir).
    t1a = np.asarray(driver.T.a).transpose(1, 0)
    t1b = np.asarray(driver.T.b).transpose(1, 0)
    n_oa, n_va = t1a.shape
    n_ob, n_vb = t1b.shape

    # T2: ccpy stores (vir, vir, occ, occ); transpose to (occ, vir, occ, vir).
    t2aa = np.asarray(driver.T.aa).transpose(2, 0, 3, 1)
    t2bb = np.asarray(driver.T.bb).transpose(2, 0, 3, 1)
    t2ab = np.asarray(driver.T.ab).transpose(2, 0, 3, 1)

    ci2aa = _ccpy_antisym_2(t2aa + 2.0 * np.einsum("ia,jb->iajb", t1a, t1a))
    ci2bb = _ccpy_antisym_2(t2bb + 2.0 * np.einsum("ia,jb->iajb", t1b, t1b))
    ci2ab = t2ab + np.einsum("ia,jb->iajb", t1a, t1b)

    amps: dict = {
        "ci1a": t1a,
        "ci1b": t1b,
        "ci2aa": ci2aa,
        "ci2ab": ci2ab,
        "ci2bb": ci2bb,
    }

    if order < 3:
        return amps

    # T3: ccpy stores (vir, vir, vir, occ, occ, occ); transpose to iajbkc.
    if order_cc >= 3:
        t3aaa = np.asarray(driver.T.aaa).transpose(3, 0, 4, 1, 5, 2)
        t3aab = np.asarray(driver.T.aab).transpose(3, 0, 4, 1, 5, 2)
        t3abb = np.asarray(driver.T.abb).transpose(3, 0, 4, 1, 5, 2)
        t3bbb = np.asarray(driver.T.bbb).transpose(3, 0, 4, 1, 5, 2)
    else:
        t3aaa = np.zeros((n_oa, n_va, n_oa, n_va, n_oa, n_va))
        t3aab = np.zeros((n_oa, n_va, n_oa, n_va, n_ob, n_vb))
        t3abb = np.zeros((n_oa, n_va, n_ob, n_vb, n_ob, n_vb))
        t3bbb = np.zeros((n_ob, n_vb, n_ob, n_vb, n_ob, n_vb))

    ci3aaa = _ccpy_antisym_3(
        _ccpy_sym_pairs_3(
            t3aaa
            + 9.0 * np.einsum("iajb,kc->iajbkc", t2aa, t1a)
            + 6.0 * np.einsum("ia,jb,kc->iajbkc", t1a, t1a, t1a)
        )
    )
    ci3bbb = _ccpy_antisym_3(
        _ccpy_sym_pairs_3(
            t3bbb
            + 9.0 * np.einsum("iajb,kc->iajbkc", t2bb, t1b)
            + 6.0 * np.einsum("ia,jb,kc->iajbkc", t1b, t1b, t1b)
        )
    )

    ci3aab_raw = (
        t3aab
        + np.einsum("iajb,kc->iajbkc", t2aa, t1b)
        + 4.0 * np.einsum("ia,jbkc->iajbkc", t1a, t2ab)
        + 2.0 * np.einsum("ia,jb,kc->iajbkc", t1a, t1a, t1b)
    )
    ci3aab = 0.5 * (ci3aab_raw + ci3aab_raw.transpose(2, 3, 0, 1, 4, 5))
    ci3aab = 0.5 * (ci3aab - ci3aab.transpose(0, 3, 2, 1, 4, 5))

    ci3abb_raw = (
        t3abb
        + np.einsum("ia,jbkc->iajbkc", t1a, t2bb)
        + 4.0 * np.einsum("iajb,kc->iajbkc", t2ab, t1b)
        + 2.0 * np.einsum("ia,jb,kc->iajbkc", t1a, t1b, t1b)
    )
    ci3abb = 0.5 * (ci3abb_raw + ci3abb_raw.transpose(0, 1, 4, 5, 2, 3))
    ci3abb = 0.5 * (ci3abb - ci3abb.transpose(0, 1, 2, 5, 4, 3))

    amps.update(
        {
            "ci3aaa": ci3aaa,
            "ci3aab": ci3aab,
            "ci3abb": ci3abb,
            "ci3bbb": ci3bbb,
        }
    )

    if order < 4:
        return amps

    # T4: ccpy stores (vir, vir, vir, vir, occ, occ, occ, occ);
    # transpose to iajbkcld.
    if order_cc >= 4:
        t4aaaa = np.asarray(driver.T.aaaa).transpose(4, 0, 5, 1, 6, 2, 7, 3)
        t4aaab = np.asarray(driver.T.aaab).transpose(4, 0, 5, 1, 6, 2, 7, 3)
        t4aabb = np.asarray(driver.T.aabb).transpose(4, 0, 5, 1, 6, 2, 7, 3)
        t4abbb = np.asarray(driver.T.abbb).transpose(4, 0, 5, 1, 6, 2, 7, 3)
        t4bbbb = np.asarray(driver.T.bbbb).transpose(4, 0, 5, 1, 6, 2, 7, 3)
    else:
        t4aaaa = np.zeros((n_oa, n_va, n_oa, n_va, n_oa, n_va, n_oa, n_va))
        t4aaab = np.zeros((n_oa, n_va, n_oa, n_va, n_oa, n_va, n_ob, n_vb))
        t4aabb = np.zeros((n_oa, n_va, n_oa, n_va, n_ob, n_vb, n_ob, n_vb))
        t4abbb = np.zeros((n_oa, n_va, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb))
        t4bbbb = np.zeros((n_ob, n_vb, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb))

    ci4aaaa = _ccpy_antisym_4(
        _ccpy_sym_pairs_4(
            t4aaaa
            + 16.0 * np.einsum("ia,jbkcld->iajbkcld", t1a, t3aaa)
            + 18.0 * np.einsum("iajb,kcld->iajbkcld", t2aa, t2aa)
            + 72.0 * np.einsum("ia,jb,kcld->iajbkcld", t1a, t1a, t2aa)
            + 24.0 * np.einsum("ia,jb,kc,ld->iajbkcld", t1a, t1a, t1a, t1a)
        )
    )
    ci4bbbb = _ccpy_antisym_4(
        _ccpy_sym_pairs_4(
            t4bbbb
            + 16.0 * np.einsum("ia,jbkcld->iajbkcld", t1b, t3bbb)
            + 18.0 * np.einsum("iajb,kcld->iajbkcld", t2bb, t2bb)
            + 72.0 * np.einsum("ia,jb,kcld->iajbkcld", t1b, t1b, t2bb)
            + 24.0 * np.einsum("ia,jb,kc,ld->iajbkcld", t1b, t1b, t1b, t1b)
        )
    )

    ci4aaab = (
        t4aaab
        + np.einsum("iajbkc,ld->iajbkcld", t3aaa, t1b)
        + 9.0 * np.einsum("ia,jbkcld->iajbkcld", t1a, t3aab)
        + 9.0 * np.einsum("iajb,kcld->iajbkcld", t2aa, t2ab)
        + 9.0 * np.einsum("iajb,kc,ld->iajbkcld", t2aa, t1a, t1b)
        + 18.0 * np.einsum("ia,jb,kcld->iajbkcld", t1a, t1a, t2ab)
        + 6.0 * np.einsum("ia,jb,kc,ld->iajbkcld", t1a, t1a, t1a, t1b)
    )
    ci4aaab = (1 / 6) * (
        ci4aaab
        + ci4aaab.transpose(0, 1, 4, 5, 2, 3, 6, 7)
        + ci4aaab.transpose(2, 3, 4, 5, 0, 1, 6, 7)
        + ci4aaab.transpose(2, 3, 0, 1, 4, 5, 6, 7)
        + ci4aaab.transpose(4, 5, 0, 1, 2, 3, 6, 7)
        + ci4aaab.transpose(4, 5, 2, 3, 0, 1, 6, 7)
    )
    ci4aaab = (1 / 6) * (
        ci4aaab
        - ci4aaab.transpose(0, 1, 2, 5, 4, 3, 6, 7)
        + ci4aaab.transpose(0, 3, 2, 5, 4, 1, 6, 7)
        - ci4aaab.transpose(0, 3, 2, 1, 4, 5, 6, 7)
        + ci4aaab.transpose(0, 5, 2, 1, 4, 3, 6, 7)
        - ci4aaab.transpose(0, 5, 2, 3, 4, 1, 6, 7)
    )

    ci4abbb = (
        t4abbb
        + 9.0 * np.einsum("iajbkc,ld->iajbkcld", t3abb, t1b)
        + np.einsum("ia,jbkcld->iajbkcld", t1a, t3bbb)
        + 9.0 * np.einsum("iajb,kcld->iajbkcld", t2ab, t2bb)
        + 18.0 * np.einsum("iajb,kc,ld->iajbkcld", t2ab, t1b, t1b)
        + 9.0 * np.einsum("ia,jb,kcld->iajbkcld", t1a, t1b, t2bb)
        + 6.0 * np.einsum("ia,jb,kc,ld->iajbkcld", t1a, t1b, t1b, t1b)
    )
    ci4abbb = (1 / 6) * (
        ci4abbb
        + ci4abbb.transpose(0, 1, 2, 3, 6, 7, 4, 5)
        + ci4abbb.transpose(0, 1, 4, 5, 6, 7, 2, 3)
        + ci4abbb.transpose(0, 1, 4, 5, 2, 3, 6, 7)
        + ci4abbb.transpose(0, 1, 6, 7, 2, 3, 4, 5)
        + ci4abbb.transpose(0, 1, 6, 7, 4, 5, 2, 3)
    )
    ci4abbb = (1 / 6) * (
        ci4abbb
        - ci4abbb.transpose(0, 1, 2, 3, 4, 7, 6, 5)
        + ci4abbb.transpose(0, 1, 2, 5, 4, 7, 6, 3)
        - ci4abbb.transpose(0, 1, 2, 5, 4, 3, 6, 7)
        + ci4abbb.transpose(0, 1, 2, 7, 4, 3, 6, 5)
        - ci4abbb.transpose(0, 1, 2, 7, 4, 5, 6, 3)
    )

    ci4aabb = (
        t4aabb
        + 4.0 * np.einsum("iajbkc,ld->iajbkcld", t3aab, t1b)
        + 4.0 * np.einsum("ia,jbkcld->iajbkcld", t1a, t3abb)
        + np.einsum("iajb,kcld->iajbkcld", t2aa, t2bb)
        + 8.0 * np.einsum("iakc,jbld->iajbkcld", t2ab, t2ab)
        + 2.0 * np.einsum("ia,jb,kcld->iajbkcld", t1a, t1a, t2bb)
        + 2.0 * np.einsum("iajb,kc,ld->iajbkcld", t2aa, t1b, t1b)
        + 16.0 * np.einsum("ia,kc,jbld->iajbkcld", t1a, t1b, t2ab)
        + 4.0 * np.einsum("ia,jb,kc,ld->iajbkcld", t1a, t1a, t1b, t1b)
    )
    ci4aabb = 0.5 * (ci4aabb + ci4aabb.transpose(2, 3, 0, 1, 4, 5, 6, 7))
    ci4aabb = 0.5 * (ci4aabb + ci4aabb.transpose(0, 1, 2, 3, 6, 7, 4, 5))
    ci4aabb = 0.5 * (ci4aabb - ci4aabb.transpose(0, 3, 2, 1, 4, 5, 6, 7))
    ci4aabb = 0.5 * (ci4aabb - ci4aabb.transpose(0, 1, 2, 3, 4, 7, 6, 5))

    amps.update(
        {
            "ci4aaaa": ci4aaaa,
            "ci4aaab": ci4aaab,
            "ci4aabb": ci4aabb,
            "ci4abbb": ci4abbb,
            "ci4bbbb": ci4bbbb,
        }
    )

    return amps


def _stage_ucisd_input_from_ccpy(driver: Any, staged_mf: Any, order_cc: int) -> TrialInput:
    amps = _ccpy_t_to_c_amplitudes(driver, order=2, order_cc=order_cc)

    mol = staged_mf.mol
    S = staged_mf.get_ovlp(mol)
    frozen = staged_mf.afqmc_frozen
    Ca = np.asarray(staged_mf.mo_coeff[0])
    Cb = np.asarray(staged_mf.mo_coeff[1])
    moa = _mf_coeff_helper(Ca, Ca, S, frozen)
    mob = _mf_coeff_helper(Ca, Cb, S, frozen)

    data = {"mo_coeff_a": np.asarray(moa), "mo_coeff_b": np.asarray(mob), **amps}
    return TrialInput(kind="ucisd", data=data, frozen=staged_mf.trial_frozen, source_kind="cc")


def _stage_ucisdt_input_from_ccpy(driver: Any, staged_mf: Any, order_cc: int) -> TrialInput:
    amps = _ccpy_t_to_c_amplitudes(driver, order=3, order_cc=order_cc)

    mol = staged_mf.mol
    S = staged_mf.get_ovlp(mol)
    frozen = staged_mf.afqmc_frozen
    Ca = np.asarray(staged_mf.mo_coeff[0])
    Cb = np.asarray(staged_mf.mo_coeff[1])
    moa = _mf_coeff_helper(Ca, Ca, S, frozen)
    mob = _mf_coeff_helper(Ca, Cb, S, frozen)

    data = {"mo_coeff_a": np.asarray(moa), "mo_coeff_b": np.asarray(mob), **amps}
    return TrialInput(kind="ucisdt", data=data, frozen=staged_mf.trial_frozen, source_kind="cc")


def _stage_ucisdtq_input_from_ccpy(driver: Any, staged_mf: Any, order_cc: int) -> TrialInput:
    amps = _ccpy_t_to_c_amplitudes(driver, order=4, order_cc=order_cc)

    mol = staged_mf.mol
    S = staged_mf.get_ovlp(mol)
    frozen = staged_mf.afqmc_frozen
    Ca = np.asarray(staged_mf.mo_coeff[0])
    Cb = np.asarray(staged_mf.mo_coeff[1])
    moa = _mf_coeff_helper(Ca, Ca, S, frozen)
    mob = _mf_coeff_helper(Ca, Cb, S, frozen)

    data = {"mo_coeff_a": np.asarray(moa), "mo_coeff_b": np.asarray(mob), **amps}
    return TrialInput(kind="ucisdtq", data=data, frozen=staged_mf.trial_frozen, source_kind="cc")


def stage_from_ccpy(
    driver: Any,
    mf: Any,
    *,
    order: int = -1,
    norb_frozen_core: int | None = None,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    fcidump: Union[str, Path, Dict[str, Any], None] = None,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> StagedInputs:
    """
    Stage AFQMC inputs from a ccpy driver and a PySCF UHF mf object.

    Args:
        driver:
            ccpy solver object with converged T amplitudes (driver.T.a, driver.T.aa, etc.)
        mf:
            PySCF UHF object used for the Hamiltonian and MO coefficients.
        order:
            CI excitation level to include (2 for UCISD, 3 for UCISDT, 4 for UCISDTQ).
            Defaults to the order of the ccpy calculation.
        norb_frozen_core:
            Number of lowest core orbitals to freeze in the AFQMC Hamiltonian.
        norb_frozen:
            Backward-compatible alias for norb_frozen_core.
        chol_cut:
            Cholesky decomposition cutoff.
        fcidump:
            Optional FCIDUMP data source used to build the Hamiltonian. Accepts a parsed
            FCIDUMP dict or a path to an FCIDUMP file. If omitted, Hamiltonian staging uses
            the provided ``mf`` object.
        cache:
            Optional path to cache results. Loads existing file if present and overwrite=False.
        overwrite:
            If True, recompute even when cache exists.
        verbose:
            Print timing/info.

    Returns:
        StagedInputs with HamInput and TrialInput
        (kind='ucisd', 'ucisdt', or 'ucisdtq'), and metadata.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load(cache_path)

    t0 = time.time()

    order_cc = int(driver.operator_params["order"])
    if order == -1:
        order = order_cc
    if order > 4:
        order = 4
    if order < 2:
        raise ValueError(f"stage_from_ccpy requires order >= 2 (got {order}).")

    resolved_frozen = _resolve_stage_frozen_arg(norb_frozen_core, norb_frozen, None)
    obj = StagedMfOrCc(mf, resolved_frozen)
    staged_mf = obj.mf

    t_ham = _stage_begin("building Hamiltonian")
    ham_source = "mf"
    if fcidump is not None and int(obj.afqmc_frozen) == 0:
        ham = _stage_ham_input_from_fcidump(
            obj, fcidump=fcidump, chol_cut=chol_cut, verbose=verbose
        )
        ham_source = "fcidump"
    else:
        if fcidump is not None and int(obj.afqmc_frozen) > 0 and verbose:
            print(
                "[stage] FCIDUMP + norb_frozen>0 requested; "
                "using existing MF frozen-core Hamiltonian staging."
            )
        ham = _stage_ham_input(obj, chol_cut=chol_cut, verbose=verbose)
    _stage_end(t_ham, "Hamiltonian ready", details=f"norb={ham.norb} nchol={ham.chol.shape[0]}")

    t_trial = _stage_begin("building trial input")
    if order <= 2:
        trial = _stage_ucisd_input_from_ccpy(driver, staged_mf, order_cc)
    elif order == 3:
        trial = _stage_ucisdt_input_from_ccpy(driver, staged_mf, order_cc)
    else:
        trial = _stage_ucisdtq_input_from_ccpy(driver, staged_mf, order_cc)
    _stage_end(t_trial, "trial input ready", details=f"kind={trial.kind}")

    mol = obj.mol
    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": "cc",
        "ccpy_order": order_cc,
        "ci_order": order,
        "ham_source": ham_source,
        "frozen": _freeze_meta_value(obj.afqmc_frozen),
        "chol_cut": ham.chol_cut,
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)

    if cache_path is not None:
        dump(staged, cache_path)

    if verbose:
        dt = time.time() - t0
        print(f"[stage] done in {dt:.2f}s | norb={ham.norb} nchol={ham.chol.shape[0]}")

    return staged
