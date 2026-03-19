from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Union

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from . import driver
from .core.system import System, WalkerKind
from .ham.chol import HamChol
from .prop.afqmc_fp import make_prop_ops_fp
from .prop.blocks import block_fp as default_block
from .prop.types import QmcParamsFp
from .setup import Job, _filter_kwargs_for, _make_trial_bundle
from .staging import StagedInputs, load, stage
from .driver import QmcResult


def _make_params_fp(
    *,
    params: QmcParamsFp | None = None,
    n_traj: int | None = None,
    ene0: float | None = None,
    n_blocks: int | None = None,
    seed: int | None = None,
    dt: float | None = None,
    n_walkers: int | None = None,
    **params_kwargs: Any,
) -> QmcParamsFp:
    base = params or QmcParamsFp()

    if seed is None and params is None:
        seed = int(np.random.randint(0, int(1e9)))

    explicit: dict[str, Any] = {}
    if n_blocks is not None:
        explicit["n_blocks"] = int(n_blocks)
    if dt is not None:
        explicit["dt"] = float(dt)
    if n_walkers is not None:
        explicit["n_walkers"] = int(n_walkers)
    if seed is not None:
        explicit["seed"] = int(seed)
    if n_traj is not None:
        explicit["n_traj"] = int(n_traj)
    if ene0 is not None:
        explicit["ene0"] = float(ene0)

    merged = dict(params_kwargs)
    merged.update(explicit)

    merged = _filter_kwargs_for(QmcParamsFp, merged)

    return replace(base, **merged)


def _make_prop_fp(
    ham_data: HamChol,
    walker_kind: str,
    sys: System,
    *,
    mixed_precision: bool,
) -> Any:
    return make_prop_ops_fp(
        ham_data.basis,
        walker_kind,
        sys,
        mixed_precision=mixed_precision,
    )


@dataclass
class JobFp(Job):
    """
    A fully assembled FP-AFQMC run bundle.
    """

    def kernel(self, **driver_kwargs: Any) -> QmcResult:
        """
        Run FP-AFQMC energy driver.
        Extra kwargs are forwarded to driver.run_qmc_energy_fp (e.g. state=..., meas_ctx=...).
        """
        assert isinstance(self.params, QmcParamsFp)
        return driver.run_qmc(
            sys=self.sys,
            params=self.params, # type: ignore
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            **driver_kwargs,
        )


def setup_fp(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # staging options (used only if we need to stage)
    norb_frozen: int | ArrayLike | None = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mixed_precision: bool = True,
    # params options
    params: QmcParamsFp | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> JobFp:
    """
    Assemble a runnable AFQMC Job from either:
      - a pyscf mf/cc object,
      - StagedInputs,
      - or a path to a staged .h5 cache file.

    Basic usage:
        job = setup_fp(mf)
        job.kernel()

    Advanced usage:
        staged = stage(cc, cache="afqmc.h5")
        job = setup_fp(staged, walker_kind="restricted", mixed_precision=False, params=myparams)
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
                norb_frozen=norb_frozen if norb_frozen is not None else None,
                chol_cut=chol_cut,
                cache=cache,
                overwrite=overwrite,
                verbose=verbose,
            )

    ham = staged.ham

    match walker_kind, ham.basis, ham.nelec[0] == ham.nelec[1]:
        case None, "restricted", True:
            walker_kind = "restricted"
        case None, "restricted", False:
            walker_kind = "unrestricted"
        case None, "generalized", _:
            walker_kind = "generalized"

    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind=walker_kind)

    ham_data = HamChol(
        jnp.asarray(ham.h0), jnp.asarray(ham.h1), jnp.asarray(ham.chol), basis=ham.basis
    )

    if params_kwargs is None:
        params_kwargs = {}
    qmc_params = _make_params_fp(
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
        prop_ops = _make_prop_fp(
            ham_data,
            sys.walker_kind,
            sys=sys,
            mixed_precision=mixed_precision,
            **prop_kwargs,
        )

    if block_fn is None:
        block_fn = default_block

    return JobFp(
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
