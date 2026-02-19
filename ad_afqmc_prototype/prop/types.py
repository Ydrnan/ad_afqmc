from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Protocol

import jax
from jax.sharding import Mesh

from ..core.ops import MeasOps, TrialOps
from ..core.system import System


class PropState(NamedTuple):
    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array


@dataclass(frozen=True)
class QmcParams:
    dt: float = 0.005
    n_chunks: int = 1
    n_exp_terms: int = 6
    pop_control_damping: float = 0.1
    weight_floor: float = 1.0e-3
    weight_cap: float = 100.0
    n_prop_steps: int = 50
    shift_ema: float = 0.1
    n_eql_blocks: int = 50
    n_blocks: int = 500
    n_walkers: int = 200
    seed: int = 42
    ene0: float = 0
    n_ene_blocks: int = 3
    


class StepKernel(Protocol):

    def __call__(
        self,
        state: PropState,
        *,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState: ...


class InitPropState(Protocol):

    def __call__(
        self,
        *,
        sys: System,
        ham_data: Any,
        trial_ops: TrialOps,
        trial_data: Any,
        meas_ops: MeasOps,
        params: QmcParams,
        initial_walkers: Any | None = None,
        initial_e_estimate: jax.Array | None = None,
        rdm1: jax.Array | None = None,
        mesh: Mesh | None = None,
    ) -> PropState: ...


@dataclass(frozen=True)
class PropOps:
    init_prop_state: InitPropState
    build_prop_ctx: Callable[
        [Any, jax.Array, QmcParams], Any
    ]  # (ham_data, rdm1, params) -> prop_ctx
    step: StepKernel


@dataclass(frozen=True)
class PropOps_fp:
    init_prop_state: InitPropState
    fp_init_prop_state: InitPropState
    build_prop_ctx: Callable[
        [Any, System, jax.Array, QmcParams], Any
    ]  # (ham_data, rdm1, params) -> prop_ctx
    step: StepKernel