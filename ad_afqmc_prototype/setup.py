from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from . import driver
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc import make_prop_ops
from .prop.blocks import block as default_block
from .prop.types import QmcParams, QmcParamsBase
from .sharding import has_model_axis, replicate, shard_model_axis
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


def _make_dataclass_params(
    params_type: type[QmcParamsBase],
    *,
    params: QmcParamsBase | None = None,
    seed: int | None = None,
    **params_kwargs: Any,
) -> QmcParamsBase:
    base = params or params_type()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    merged = dict(params_kwargs)
    if seed is not None:
        merged["seed"] = int(seed)

    merged = _filter_kwargs_for(params_type, merged)

    return replace(base, **merged)


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
    explicit: dict[str, Any] = {}
    if n_eql_blocks is not None:
        explicit["n_eql_blocks"] = int(n_eql_blocks)
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    return cast(
        QmcParams,
        _make_dataclass_params(
            QmcParams,
            params=params,
            seed=seed,
            **params_kwargs,
            **explicit,
        ),
    )


def _make_prop(
    ham_data: HamChol,
    walker_kind: str,
    sys: System | None = None,
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
        return HamChol(
            replicate(jnp.asarray(ham.h0), mesh),
            replicate(ham.h1, mesh),
            shard_model_axis(ham.chol, mesh),
            basis=ham.basis,
        )

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


def _resolve_default_walker_kind(ham: Any, walker_kind: WalkerKind | None) -> WalkerKind:
    if walker_kind is None:
        return cast(WalkerKind, ham.basis)
    return walker_kind


def _compact_ham_data_for_runtime(ham_data: Any, meas_ctx: Any) -> Any:
    if not isinstance(ham_data, HamChol):
        return ham_data

    from .meas.ghf import GhfCholMeasCtx
    from .meas.rhf import RhfMeasCtx
    from .meas.uhf import UhfMeasCtx

    if isinstance(meas_ctx, (RhfMeasCtx, UhfMeasCtx, GhfCholMeasCtx)):
        chol = ham_data.chol
        if isinstance(chol, jax.Array):
            compact_chol = jax.device_put(
                jnp.zeros((chol.shape[0], 0, 0), dtype=chol.dtype),
                chol.sharding,
            )
        else:
            compact_chol = chol[:, :0, :0]
        return HamChol(
            h0=ham_data.h0,
            h1=ham_data.h1,
            chol=compact_chol,
            basis=ham_data.basis,
        )

    return ham_data


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
    _runtime_prop_ctx: Any = field(default=None, init=False, repr=False)
    _runtime_meas_ctx: Any = field(default=None, init=False, repr=False)
    _runtime_state: Any = field(default=None, init=False, repr=False)

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParams
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver.run_qmc_energy)

    def _prepare_runtime(
        self,
        *,
        state: Any = None,
        meas_ctx: Any = None,
        prop_ctx: Any = None,
    ) -> tuple[Any, Any, Any]:
        if prop_ctx is None:
            prop_ctx = self._runtime_prop_ctx
        if meas_ctx is None:
            meas_ctx = self._runtime_meas_ctx
        if state is None:
            state = self._runtime_state

        if prop_ctx is not None and meas_ctx is not None and state is not None:
            return state, meas_ctx, prop_ctx

        ham_data_full = self.ham_data

        if prop_ctx is None:
            prop_ctx = self.prop_ops.build_prop_ctx(
                ham_data_full,
                self.trial_ops.get_rdm1(self.trial_data),
                self.params,
            )
        if meas_ctx is None:
            meas_ctx = self.meas_ops.build_meas_ctx(ham_data_full, self.trial_data)
        if state is None:
            state = self.prop_ops.init_prop_state(
                sys=self.sys,
                ham_data=ham_data_full,
                trial_ops=self.trial_ops,
                trial_data=self.trial_data,
                meas_ops=self.meas_ops,
                params=self.params,
                mesh=self.mesh,
            )

        if self.params_cls is QmcParams:
            ham_data_runtime = _compact_ham_data_for_runtime(ham_data_full, meas_ctx)
            if ham_data_runtime is not ham_data_full:
                self.ham_data = ham_data_runtime

        self._runtime_prop_ctx = prop_ctx
        self._runtime_meas_ctx = meas_ctx
        self._runtime_state = state
        return state, meas_ctx, prop_ctx

    def kernel(self, **driver_kwargs: Any):
        """
        Run AFQMC energy driver.
        Extra kwargs are forwarded to driver.run_qmc_energy (e.g. state=..., meas_ctx=...).
        """
        assert isinstance(self.params, self.params_cls)
        state, meas_ctx, prop_ctx = self._prepare_runtime(
            state=driver_kwargs.get("state"),
            meas_ctx=driver_kwargs.get("meas_ctx"),
            prop_ctx=driver_kwargs.get("prop_ctx"),
        )
        driver_kwargs["state"] = state
        driver_kwargs["meas_ctx"] = meas_ctx
        driver_kwargs["prop_ctx"] = prop_ctx
        driver_kwargs.setdefault("mesh", self.mesh)
        return self.driver_fn(
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


def _assemble_job(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    norb_frozen: int | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    params: QmcParamsBase | None = None,
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
    params_builder: Callable[..., QmcParamsBase],
    prop_builder: Callable[..., Any],
    default_block_fn: Callable[..., Any],
    job_cls: type[Job],
    walker_kind_resolver: Callable[[Any, WalkerKind | None], WalkerKind],
) -> Job:
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

    resolved_walker_kind = walker_kind_resolver(ham, walker_kind)
    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=resolved_walker_kind)

    qmc_params = params_builder(params=params, **(params_kwargs or {}))

    if trial_data is None or trial_ops is None or meas_ops is None:
        td, to, mo = _make_trial_bundle(sys, staged, mixed_precision)
        trial_data = td if trial_data is None else trial_data
        trial_ops = to if trial_ops is None else trial_ops
        meas_ops = mo if meas_ops is None else meas_ops

    if prop_ops is None:
        prop_ops = prop_builder(
            ham_data,
            sys.walker_kind,
            sys=sys,
            mixed_precision=mixed_precision,
            **(prop_kwargs or {}),
        )

    if block_fn is None:
        block_fn = default_block_fn

    return job_cls(
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
    return _assemble_job(
        obj_or_staged,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        cache=cache,
        overwrite=overwrite,
        verbose=verbose,
        walker_kind=walker_kind,
        mesh=mesh,
        mixed_precision=mixed_precision,
        params=params,
        trial_data=trial_data,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
        params_kwargs=params_kwargs,
        prop_kwargs=prop_kwargs,
        params_builder=_make_params,
        prop_builder=_make_prop,
        default_block_fn=default_block,
        job_cls=Job,
        walker_kind_resolver=_resolve_default_walker_kind,
    )
