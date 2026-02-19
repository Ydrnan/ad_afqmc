from __future__ import annotations

import time
from functools import partial
from pprint import pprint
from typing import Any, Callable
from dataclasses import dataclass, replace

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .core.ops import MeasOps, TrialOps
from .core.system import System
from .prop.blocks import BlockFn
from .prop.types import PropOps, PropState, QmcParams, PropOps_fp
from .stat_utils import blocking_analysis_ratio, reject_outliers
from .walkers import stochastic_reconfiguration

print = partial(print, flush=True)


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                trial_data=trial_data,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ctx,
                prop_ops=prop_ops,
                prop_ctx=prop_ctx,
            )
            return state, (obs.scalars["energy"], obs.scalars["weight"])

        stateN, (e, w) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w

    return run_blocks

def make_run_blocks_fp(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps_fp,
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks","n_ene_blocks"))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
        n_ene_blocks: int,
    ):
        def one_block(state, n):
            state, obs = block_fn(
                    state,
                    sys=sys,
                    params=params,
                    ham_data=ham_data,
                    trial_data=trial_data,
                    trial_ops=trial_ops,
                    meas_ops=meas_ops,
                    meas_ctx=meas_ctx,
                    prop_ops=prop_ops,
                    prop_ctx=prop_ctx,
                )
            return state, (obs.scalars["energy"], obs.scalars["weight"], obs.scalars["overlap"], obs.scalars["abs_overlap"])
        stateN, (e, w, ov, abs_ov) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w, ov, abs_ov

    return run_blocks

def run_qmc_energy(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    equilibration blocks then sampling blocks.

    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    # build ctx
    prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )
    
    
    if mesh is None or mesh.size == 1:
        block_fn_sr = block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_w_eq = []
    block_e_eq.append(state.e_estimate)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'E_blk':>14s}  "
            f"{'W':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(e_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )
    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = stats["mu"]
        se = stats["se_star"]
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{(f'{se:10.3e}' if se is not None else ' ' * 10)}  "
            f"{float(e_chunk_avg):16.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if se is not None and se <= target_error and target_error > 0.0:
            print(f"\nTarget error {target_error:.3e} reached at block {start + n}.")
            break
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)

    data_clean, _ = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])

    return mean, err, block_e_all, block_w_all

def run_qmc_energy_fp(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps_fp,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array]:
    """

    Returns:
      (mean_energy, stderr, block_energies, block_weights)
    """
    print("Starting QMC driver...")
    print(f"Parameters:")
    pprint(params)
    print("")
    # build ctx
    prop_ctx = prop_ops.build_prop_ctx(ham_data,sys, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )
    
    
    # if mesh is None or mesh.size == 1:
    #     block_fn_sr = block_fn
    # else:
    #     data_sh = NamedSharding(mesh, P("data"))
    #     sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
    #     block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    block_fn_sr = block_fn

    run_blocks = make_run_blocks_fp(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()
    t_mark = t0

    # sampling
    print("\nSampling:\n")
    #print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 1
    print_every = 1
    block_e_all = np.zeros((params.n_ene_blocks, params.n_blocks+1)) +0.0j
    block_w_all = np.zeros((params.n_ene_blocks, params.n_blocks+1)) +0.0j  
    total_sign =  np.ones((params.n_ene_blocks, params.n_blocks+1)) + 0.0j
    block_e_all[:,0] = jnp.array(state.e_estimate)
    block_w_all[:,0] = jnp.sum(state.weights)
    total_sign[:,0] = jnp.sum(state.overlaps) / (jnp.sum(jnp.abs(state.overlaps)))
    chunk = print_every
    for i in range(params.n_ene_blocks):
        block_e_s = []
        block_w_s = []
        print("Trajectory count", i)
        if i > 0 :
            params = replace( params, seed = params.seed + i)
            state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )
        for j,start in enumerate(range(0, params.n_blocks+1, chunk)):
            n = min(chunk, params.n_blocks - start)
            state, e_chunk, w_chunk, ov_chunk, abs_ov_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
            n_ene_blocks=params.n_ene_blocks,
            )
            block_e_s.extend(e_chunk.tolist())
            block_w_s.extend(w_chunk.tolist())
            block_e_all[i,(j*n)+1:(j*n+len(e_chunk)+1)] = np.array(e_chunk.tolist())
            block_w_all[i,(j*n)+1:(j*n+len(w_chunk)+1)] = np.array(w_chunk.tolist())   
            sign = ov_chunk / abs_ov_chunk
            total_sign[i,(j*n)+1:(j*n+len(sign)+1)] = np.array(sign.tolist())
            mean_energies = jnp.sum(block_e_all[:i+1]*block_w_all[:i+1],axis=0)/jnp.sum(block_w_all[:i+1],axis=0)
            mean_sign = jnp.sum(total_sign[:i+1]*block_w_all[:i+1],axis=0)/jnp.sum(block_w_all[:i+1],axis=0)
            if i == 0:
                error = jnp.zeros_like(mean_energies)
            else:
                error = jnp.std(block_e_all[:i+1],axis=0)/jnp.sqrt(i)

            timer = params.dt*params.n_prop_steps*chunk*jnp.arange(params.n_blocks+1)
            print(
            f"{(timer[j]):14.4f} "
            f"{(mean_energies[j*chunk].real):14.10f}  "
            f"{(error[j*chunk].real):10.3e}  "
            f"{(mean_sign[j*chunk].real):10.2f}"
        )
            elapsed = time.perf_counter() - t0
        
        print(f"Wall time :{elapsed:12.1f} s\n")
        
        

    return  block_e_all, block_w_all 
