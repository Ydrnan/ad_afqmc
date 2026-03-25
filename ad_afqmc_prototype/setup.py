from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Union

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from . import driver
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc import make_prop_ops
from .prop.blocks import block as default_block
from .prop.types import QmcParams, QmcParamsBase
from .sharding import has_model_axis, shard_ham_data
from .staging import StagedInputs, load, stage


def _filter_kwargs_for(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only those accepted by callable_obj's signature.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _make_params(
    *,
    params: QmcParams | None = None,
    n_eql_blocks: int | None = None,
    n_blocks: int | None = None,
    seed: int | None = None,
    dt: float | None = None,
    n_walkers: int | None = None,
    **params_kwargs: Any,
) -> QmcParams:
    base = params or QmcParams()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    explicit: dict[str, Any] = {}
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

    merged = _filter_kwargs_for(QmcParams, merged)

    return replace(base, **merged)


def _make_prop(
    ham_data: HamChol,
    walker_kind: str,
    *,
    mixed_precision: bool,
) -> Any:
    return make_prop_ops(
        ham_data.basis,
        walker_kind,
        mixed_precision=mixed_precision,
    )


def _make_ham_data(ham: Any, mesh: Mesh | None) -> HamChol:
    if mesh is not None and mesh.size > 1 and has_model_axis(mesh):
        ham_data = HamChol(ham.h0, ham.h1, ham.chol, basis=ham.basis)
        return shard_ham_data(ham_data, mesh)

    return HamChol(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1),
        jnp.asarray(ham.chol),
        basis=ham.basis,
    )


def _resolve_staged_and_ham_data(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    norb_frozen: int | None,
    chol_cut: float,
    cache: Union[str, Path] | None,
    overwrite: bool,
    verbose: bool,
    mesh: Mesh | None,
) -> tuple[StagedInputs, HamChol]:
    staged: StagedInputs
    if isinstance(obj_or_staged, StagedInputs):
        staged = obj_or_staged
        return staged, _make_ham_data(staged.ham, mesh)

    p = (
        Path(obj_or_staged).expanduser().resolve()
        if isinstance(obj_or_staged, (str, Path))
        else None
    )
    if p is not None and p.exists():
        staged = load(p)
        return staged, _make_ham_data(staged.ham, mesh)

    staged = stage(
        obj_or_staged,
        norb_frozen=norb_frozen if norb_frozen is not None else 0,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
    )
    return staged, _make_ham_data(staged.ham, mesh)


def _make_trial_bundle(
    sys: System, staged: StagedInputs, mixed_precision: bool
) -> tuple[Any, Any, Any]:
    """
    Return (trial_data, trial_ops, meas_ops)
    """
    tr = staged.trial
    data = tr.data

    kind = tr.kind.lower()

    if kind == "rhf":
        from .meas.rhf import make_rhf_meas_ops
        from .trial.rhf import make_rhf_trial_data, make_rhf_trial_ops

        trial_data = make_rhf_trial_data(data, sys)
        trial_ops = make_rhf_trial_ops(sys=sys)
        meas_ops = make_rhf_meas_ops(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "uhf":
        from .meas.uhf import make_uhf_meas_ops
        from .trial.uhf import make_uhf_trial_data, make_uhf_trial_ops

        trial_data = make_uhf_trial_data(data, sys)
        trial_ops = make_uhf_trial_ops(sys=sys)
        meas_ops = make_uhf_meas_ops(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "ghf":
        from .meas.ghf import make_ghf_meas_ops_chol
        from .trial.ghf import make_ghf_trial_data, make_ghf_trial_ops

        trial_data = make_ghf_trial_data(data, sys=sys)
        trial_ops = make_ghf_trial_ops(sys=sys)
        meas_ops = make_ghf_meas_ops_chol(sys=sys)
        return trial_data, trial_ops, meas_ops

    if kind == "cisd":
        from .meas.cisd import make_cisd_meas_ops
        from .trial.cisd import make_cisd_trial_data, make_cisd_trial_ops

        trial_data = make_cisd_trial_data(data, sys)
        trial_ops = make_cisd_trial_ops(sys=sys)
        meas_ops = make_cisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        return trial_data, trial_ops, meas_ops

    if kind == "ucisd":
        from .meas.ucisd import make_ucisd_meas_ops
        from .trial.ucisd import make_ucisd_trial_data, make_ucisd_trial_ops

        trial_data = make_ucisd_trial_data(data, sys)
        trial_ops = make_ucisd_trial_ops(sys=sys)
        meas_ops = make_ucisd_meas_ops(sys=sys, mixed_precision=mixed_precision)
        return trial_data, trial_ops, meas_ops

    if kind == "gcisd":
        from .meas.gcisd import make_gcisd_meas_ops
        from .trial.gcisd import make_gcisd_trial_data, make_gcisd_trial_ops

        trial_data = make_gcisd_trial_data(data, sys)
        trial_ops = make_gcisd_trial_ops(sys=sys)
        meas_ops = make_gcisd_meas_ops(sys=sys)
        return trial_data, trial_ops, meas_ops

    raise ValueError(f"Unsupported TrialInput.kind: {tr.kind!r}")


@dataclass
class Job:
    """
    A fully assembled AFQMC run bundle.
    """

    staged: StagedInputs
    sys: System
    params: QmcParamsBase
    ham_data: Any
    trial_data: Any
    trial_ops: Any
    meas_ops: Any
    prop_ops: Any
    block_fn: Callable[..., Any]
    mesh: Mesh | None = None

    def kernel(self, **driver_kwargs: Any):
        """
        Run AFQMC energy driver.
        Extra kwargs are forwarded to driver.run_qmc_energy (e.g. state=..., meas_ctx=...).
        """
        assert isinstance(self.params, QmcParams)
        driver_kwargs.setdefault("mesh", self.mesh)
        return driver.run_qmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            **driver_kwargs,
        )


def setup(
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
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    # params options
    params: QmcParams | None = None,
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
        job = setup(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
        job.kernel()
    """
    staged, ham_data = _resolve_staged_and_ham_data(
        obj_or_staged,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
        mesh=mesh,
    )
    ham = staged.ham

    if walker_kind is None:
        walker_kind = ham.basis

    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)

    if params_kwargs is None:
        params_kwargs = {}
    qmc_params = _make_params(
        params=params,
        **params_kwargs,
    )

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle(sys, staged, mixed_precision)
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
        mesh=mesh,
    )
