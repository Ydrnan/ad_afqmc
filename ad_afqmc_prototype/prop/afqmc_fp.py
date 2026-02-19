from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..sharding import shard_prop_state
from ..walkers import init_walkers
from .chol_afqmc_ops import  TrotterOps, make_trotter_ops
from .afqmc import init_prop_state, fp_init_prop_state
from .fp_chol_afqmc_ops import FpCholAfqmcCtx, _build_prop_ctx_fp
from .types import PropOps_fp, PropState, QmcParams

def afqmc_step_fp(
    state: PropState,
    sys: System,
    *,
    params: QmcParams,
    ham_data: HamChol,
    trial_data: Any,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: FpCholAfqmcCtx,
    meas_ctx: Any,
) -> PropState:
    # jax.debug.print("state a {}", state.rng_key)
    key, subkey = jax.random.split(state.rng_key)
    # jax.debug.print("key for fields {b}", b=subkey)
    nw = wk.n_walkers(state.walkers)
    fields = jax.random.normal(subkey, (nw, ham_data.chol.shape[0]))
#    jax.debug.print("fields {a}", a=fields)
    wk_kind = sys.walker_kind.lower()
    assert wk_kind in [
            "restricted",
            "unrestricted",
        ], "Free propagation is only implemented for restricted and unrestricted walkers."
    if wk_kind == "unrestricted":
        shift_term = jnp.einsum("wg,sg->sw", fields, prop_ctx.mf_fp_shift)
        constants = jnp.einsum(
                "sw,s->sw",
                jnp.exp(-jnp.sqrt(prop_ctx.dt) * shift_term),
                jnp.exp(prop_ctx.dt * prop_ctx.h0_prop_fp),
            )
    else:
        shift_term = jnp.einsum("wg,g->w", fields, prop_ctx.mf_fp_shift)
        constants = jnp.exp(-jnp.sqrt(prop_ctx.dt) * shift_term) * jnp.exp(
                prop_ctx.dt * prop_ctx.h0_prop_fp
            )

#    jax.debug.print("walkers before prop {a}", a=state.walkers)
    walkers_new = wk.vmap_chunked(
        trotter_ops.apply_trotter, n_chunks=params.n_chunks, in_axes=(0, 0, None, None)
    )(state.walkers, fields, prop_ctx, 10)
#    jax.debug.print("walkers after prop {a}", a=walkers_new)
#    jax.debug.print("constants {a}", a=constants)
    walkers_new = wk.multiply_constants(walkers_new,constants)
#    jax.debug.print("walkers after multiplying constants {a}", a=walkers_new)
    q, norms = wk.orthogonalize(walkers_new, wk_kind)
#    jax.debug.print("walkers after orthogonalization {a}", a=q)
#    jax.debug.print("norms after orthogonalization {a}", a=norms)
    weights_new = state.weights*norms.real
#    jax.debug.print("walkers weight after orthogonalization {a}", a=weights_new)
    key , subkey = jax.random.split(key)
    zeta = jax.random.uniform(subkey)
    # jax.debug.print("zeta subkey {a}", a=subkey)
    # jax.debug.print("updated key after prop {}", key)
#    jax.debug.print("zeta {a}", a=zeta)
#    jax.debug.print("walkers weight after prop {a}", a=weights_new)

    walker_sr, weight_sr = wk.stochastic_reconfiguration(q,weights_new,zeta,wk_kind)
#    jax.debug.print("walkers after sr {a}", a=walker_sr)    
#    weight_sr /= norms.real

    return PropState(walkers=walker_sr,
        weights=weight_sr,
        overlaps=state.overlaps,
        rng_key=key,
        pop_control_ene_shift=state.pop_control_ene_shift,
        e_estimate=state.e_estimate,
        node_encounters=state.node_encounters,
        )

def make_prop_ops_fp(ham_basis: str, walker_kind: str, sys:System, mixed_precision=False) -> PropOps_fp:
    trotter_ops = make_trotter_ops(
        ham_basis, walker_kind, mixed_precision=mixed_precision
    )
        
    def step_fp(
        state: PropState,
        *,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropState:
        return afqmc_step_fp(
            state,
            sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,
        )
    
    def build_prop_ctx_fp(
        ham_data: Any, sys: System, rdm1: jax.Array, params: QmcParams
    ) -> FpCholAfqmcCtx:
            return _build_prop_ctx_fp(
            ham_data,
            sys,
            rdm1,
            params.dt,
            params.ene0,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
        )
        

    return PropOps_fp(
            init_prop_state=init_prop_state, fp_init_prop_state=fp_init_prop_state,build_prop_ctx=build_prop_ctx_fp, step=step_fp
    )
