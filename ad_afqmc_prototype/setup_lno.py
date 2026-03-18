import numpy as np
from numpy.typing import NDArray, ArrayLike
import jax.numpy as jnp
from pathlib import Path
from typing import Any, Callable, Union
from dataclasses import replace

from pyscf import mcscf, ao2mo

from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .setup import Job, _filter_kwargs_for, _make_prop
from .prop.types import QmcParamsLno
from .prop.blocks import block as default_block
from .staging import StagedInputs, load, stage, HamInput, StagedMfOrCc, modified_cholesky


def _make_params(
    *,
    params: QmcParamsLno | None = None,
    prjlo: NDArray | None = None,
    n_eql_blocks: int | None = None,
    n_blocks: int | None = None,
    seed: int | None = None,
    dt: float | None = None,
    n_walkers: int | None = None,
    **params_kwargs: Any,
) -> QmcParamsLno:
    base = params or QmcParamsLno()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    explicit: dict[str, Any] = {}
    if prjlo is not None:
        explicit["prjlo"] = np.asarray(prjlo)
    if n_eql_blocks is not None:
        explicit["n_eql_blocks"] = int(n_eql_blocks)
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    if seed is not None:
        explicit["seed"] = int(seed)

    merged = dict(params_kwargs)
    merged.update(explicit)

    merged = _filter_kwargs_for(QmcParamsLno, merged)

    params = replace(base, **merged)

    if params.prjlo is None:
        raise ValueError("The parameter 'prjlo' must be set.")

    return params


def setup_lno(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mixed_precision: bool = True,
    # params options
    params: QmcParamsLno | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> Job:
    """
    Assemble a runnable AFQMC Job from either:
      - a pyscf mf/cc object,
      - StagedInputs,
      - or a path to a staged .h5 cache file.

    Basic usage:
        job = setup_lno(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup_lno(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
        job.kernel()
    """
    staged: StagedInputs
    if isinstance(obj_or_staged, StagedInputs):
        staged = obj_or_staged
    else:
        p = (
            Path(obj_or_staged).expanduser().resolve()
            if isinstance(obj_or_staged, (str, Path))
            else None
        )
        if p is not None and p.exists():
            staged = load(p)
        else:
            staged = stage(
                obj_or_staged,
                norb_frozen=norb_frozen if norb_frozen is not None else 0,
                chol_cut=chol_cut,
                cache=cache,
                overwrite=overwrite,
                verbose=verbose,
            )

    ham = staged.ham

    if walker_kind is None:
        walker_kind = ham.basis

    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)

    ham_data = HamChol(
        jnp.asarray(ham.h0), jnp.asarray(ham.h1), jnp.asarray(ham.chol), basis=ham.basis
    )

    if params_kwargs is None:
        params_kwargs = {}
    qmc_params = _make_params(
        params=params,
        **params_kwargs,
    )

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle(sys, staged, qmc_params, mixed_precision)
        trial_data = td if trial_data is None else trial_data
        trial_ops = to if trial_ops is None else trial_ops
        meas_ops = mo if meas_ops is None else meas_ops

    if prop_ops is None:
        if prop_kwargs is None:
            prop_kwargs = {}
        prop_ops = _make_prop(
            ham_data,
            sys.walker_kind,
            mixed_precision=mixed_precision,
            **prop_kwargs,
        )

    if block_fn is None:
        block_fn = default_block

    return Job(
        staged=staged,
        sys=sys,
        params=qmc_params,
        ham_data=ham_data,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
    )


def _make_trial_bundle(
    sys: System, staged: StagedInputs, params: QmcParamsLno, mixed_precision: bool
) -> tuple[Any, Any, Any]:
    """
    Return (trial_data, trial_ops, meas_ops)
    """
    tr = staged.trial
    data = tr.data

    kind = tr.kind.lower()

    if kind == "rhf":
        from .meas.rhf import make_lno_rhf_meas_ops
        from .trial.rhf import make_rhf_trial_ops, make_rhf_trial_data

        trial_data = make_rhf_trial_data(data, sys)
        trial_ops = make_rhf_trial_ops(sys=sys)
        meas_ops = make_lno_rhf_meas_ops(sys=sys, params=params)
        return trial_data, trial_ops, meas_ops

    raise ValueError(f"Unsupported TrialInput.kind: {tr.kind!r}")


def build_ham(
    obj: Any,
    *,
    norb_frozen: ArrayLike,
    chol_cut: float,
) -> HamInput:
    obj = StagedMfOrCc(obj, norb_frozen)
    mf = obj.mf.mf
    mol = mf.mol

    norb = obj.norb
    norb_frozen = np.array(obj.norb_frozen)
    basis_coeff = mf.mo_coeff

    nelec_frozen = 2 * np.sum(norb_frozen < mol.nelec[0])
    nact = basis_coeff.shape[1] - norb_frozen.size
    nelec_act = mol.nelectron - nelec_frozen
    mc = mcscf.CASSCF(mf, nact, nelec_act)
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    h1, h0 = mc.get_h1eff()
    act = [i for i in range(norb) if i not in norb_frozen]
    e = ao2mo.kernel(mf.mol, mf.mo_coeff[:, act])  # , compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)
    chol = chol.reshape((-1, nact, nact))

    ham = HamInput(
        h0=float(h0),
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        norb_frozen=norb_frozen,
        source_kind=obj.source,
        basis="restricted",
    )

    return ham
