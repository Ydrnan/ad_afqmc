from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..ham.chol import HamChol
from .utils import taylor_expm_action
from ..prop.chol_afqmc_ops import _mf_shifts, _build_exp_h1_half_from_h1
from ..core.system import System

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FpCholAfqmcCtx:
    dt: jax.Array
    sqrt_dt: jax.Array
    exp_h1_half: jax.Array  # (n,n) or (ns,ns)
    mf_shifts: jax.Array  # (n_fields,)
    h0_prop: jax.Array  # scalar
    chol_flat: jax.Array  # (n_fields, n*n)
    mf_fp_shift: jax.Array  
    h0_prop_fp: jax.Array 
    norb: int

    def tree_flatten(self):
        return (
            self.dt,
            self.sqrt_dt,
            self.exp_h1_half,
            self.mf_shifts,
            self.h0_prop,
            self.chol_flat,
            self.mf_fp_shift,
            self.h0_prop_fp,
        ), (self.norb,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        dt, sqrt_dt, exp_h1_half, mf_shifts, h0_prop, chol_flat, mf_fp_shift, h0_prop_fp = children
        (norb,) = aux

        return cls(
            dt=dt,
            sqrt_dt=sqrt_dt,
            exp_h1_half=exp_h1_half,
            mf_shifts=mf_shifts,
            h0_prop=h0_prop,
            chol_flat=chol_flat,
            mf_fp_shift=mf_fp_shift,
            h0_prop_fp=h0_prop_fp,
            norb=norb,     
        )

def _get_h1_eff(ham_data: HamChol, mf: jax.Array) -> jax.Array:
    match ham_data.basis:
        case "restricted" | "generalized":
            v0m = 0.5 * jnp.einsum(
                "gik,gkj->ij", ham_data.chol, ham_data.chol, optimize="optimal"
            )
            mf_r = (1.0j * mf).real
            v1m = jnp.einsum("g,gik->ik", mf_r, ham_data.chol, optimize="optimal")
            h1_eff = ham_data.h1 - v0m - v1m
        case _:
            raise ValueError(f"Unknown Hamiltonian basis kind: {ham_data.basis}")

    return h1_eff

def _build_prop_ctx_fp(ham_data: HamChol,
    sys: System,                    
    rdm1: jax.Array,
    dt: float,
    ene0: jnp.dtype = jnp.float64,
    chol_flat_precision: jnp.dtype = jnp.float64,   
) -> FpCholAfqmcCtx:
    dt_a = jnp.array(dt)
    sqrt_dt = jnp.sqrt(dt_a)
    mf = _mf_shifts(ham_data, rdm1)
    h0_prop = -ham_data.h0 - 0.5 * jnp.sum(mf**2)
    
    if sys.walker_kind.lower() == "unrestricted":
        mf_fp = jnp.stack(
            (
                mf / sys.nelec[0] / 2,
                mf / sys.nelec[1] / 2,
            ) 
        )
        h0_prop_fp = jnp.stack(
            (
            0.5*(h0_prop + ene0)/sys.nelec[0],
            0.5*(h0_prop + ene0)/sys.nelec[1],
            )
        )
        
    
    if sys.walker_kind.lower() == "restricted":
        mf_fp = mf * 0.5 / sys.nelec[0]
        h0_prop_fp = 0.5*(h0_prop + ene0)/sys.nelec[0]
        h1_eff = _get_h1_eff(ham_data,mf)
        exp_h1_half = _build_exp_h1_half_from_h1(h1_eff, dt_a)


    h1_eff = _get_h1_eff(ham_data,mf)
    exp_h1_half = _build_exp_h1_half_from_h1(h1_eff,dt_a)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], -1).astype(
        chol_flat_precision
    )
    norb = ham_data.chol.shape[1]
    return FpCholAfqmcCtx(
        dt=dt_a,
        sqrt_dt=sqrt_dt,
        exp_h1_half=exp_h1_half,
        mf_shifts=mf,
        h0_prop=h0_prop,
        chol_flat=chol_flat,
        mf_fp_shift = mf_fp,
        h0_prop_fp = h0_prop_fp,
        norb=norb,     
    )
