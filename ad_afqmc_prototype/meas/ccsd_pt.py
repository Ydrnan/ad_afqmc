from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util, vmap, lax

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.ccsd_pt import CCSDpT_Trial, overlap_r
from .rhf import _half_green_from_overlap_matrix
from .rhf import force_bias_kernel_r
from .cisd import _greens_restricted, _greenp_from_green

_half_green_from_overlap_matrix = _half_green_from_overlap_matrix
force_bias_kernel_r = force_bias_kernel_r
# force_bias_kernel_u = force_bias_kernel_u
# force_bias_kernel_g = force_bias_kernel_g

def energy_kernel_r(
        walker: jax.Array,
        ham_data: HamChol, 
        meas_ctx: CCSDpTMeasCtx,
        trial_data: CCSDpT_Trial
        ) -> jax.Array:
    
    nocc = trial_data.nocc
    t1, t2 = trial_data.t1, trial_data.t2
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    green = _greens_restricted(walker, nocc)  # (nocc, norb)
    green_occ = green[:, nocc:]  # (nocc, nvir)
    greenp = _greenp_from_green(green)  # (norb, nvir)

    h0 = ham_data.h0
    h1 = ham_data.h1
    chol = ham_data.chol
    rot_chol = meas_ctx.rot_chol

    # 1 body energy
    # ref
    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # single excitations
    t1g = jnp.einsum("pt,pt->", t1, green_occ, optimize="optimal")
    e1_1_1 = 4 * t1g * hg
    gpt1 = greenp @ t1.T
    t1_green = gpt1 @ green
    e1_1_2 = -2 * jnp.einsum("ij,ij->", h1, t1_green, optimize="optimal")
    e1_1 = e1_1_1 + e1_1_2

    # double excitations
    t2g_c = jnp.einsum("ptqu,pt->qu", t2, green_occ, optimize="optimal")
    t2g_e = jnp.einsum("ptqu,pu->qt", t2, green_occ, optimize="optimal")
    t2_green_c = (greenp @ t2g_c.T) @ green
    t2_green_e = (greenp @ t2g_e.T) @ green
    t2_green = 2 * t2_green_c - t2_green_e
    t2g = 2 * t2g_c - t2g_e
    gt2g = jnp.einsum("qu,qu->", t2g, green_occ, optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("ij,ij->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    # two body energy
    # ref
    lg = jnp.einsum("gpj,pj->g", rot_chol, green, optimize="optimal")
    lg1 = jnp.einsum("gpj,qj->gpq", rot_chol, green, optimize="optimal")
    e2_0_1 = 2 * lg @ lg
    e2_0_2 = -jnp.sum(vmap(lambda x: x * x.T)(lg1))
    e2_0 = e2_0_1 + e2_0_2

    # single excitations
    e2_1_1 = 2 * e2_0 * t1g
    lt1g = jnp.einsum("gij,ij->g", chol, t1_green, optimize="optimal")
    e2_1_2 = -2 * (lt1g @ lg)
    t1g1 = t1 @ green_occ.T
    e2_1_3_1 = jnp.einsum("gpq,gqr,rp->", lg1, lg1, t1g1, optimize="optimal")
    lt1g = jnp.einsum("gip,qi->gpq", meas_ctx.lt1, green, optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gqp->", lt1g, lg1, optimize="optimal")
    e2_1_3 = e2_1_3_1 + e2_1_3_2
    e2_1 = e2_1_1 + 2 * (e2_1_2 + e2_1_3)

    # double excitations
    e2_2_1 = e2_0 * gt2g
    lt2g = jnp.einsum("gij,ij->g", chol, t2_green, optimize="optimal")
    e2_2_2_1 = -lt2g @ lg

    def scanned_fun(carry, x):
        chol_i, rot_chol_i = x
        gl_i = jnp.einsum("pj,ji->pi", green, chol_i, optimize="optimal")
        lt2_green_i = jnp.einsum("pi,ji->pj", rot_chol_i, t2_green, optimize="optimal")
        carry[0] += 0.5 * jnp.einsum("pi,pi->", gl_i, lt2_green_i, optimize="optimal")
        glgp_i = jnp.einsum("pi,it->pt", gl_i, greenp, optimize="optimal")
        l2t2_1 = jnp.einsum("pt,qu,ptqu->", glgp_i, glgp_i, t2, optimize="optimal")
        l2t2_2 = jnp.einsum("pu,qt,ptqu->", glgp_i, glgp_i, t2, optimize="optimal")
        carry[1] += 2 * l2t2_1 - l2t2_2
        return carry, 0.0

    [e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [0.0, 0.0], (chol, rot_chol))
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)

    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    e0 = h0 + e1_0 + e2_0 # h0 + <psi|(h1+h2)|phi>/<psi|phi>
    t = 2 * t1g + gt2g # <psi|(t1+t2)|phi>/<psi|phi>
    te = e1_1 + e1_2 + e2_1 + e2_2 # <psi|(t1+t2)(h1+h2)|phi>/<psi|phi>

    return [e0, t, te]

@dataclass(frozen=True)
class CCSDpTMeasCfg:
    memory_mode: str = "low"  # or Literal["low","high"]
    mixed_real_dtype: jnp.dtype = jnp.float64
    mixed_complex_dtype: jnp.dtype = jnp.complex128
    mixed_real_dtype_testing: jnp.dtype = jnp.float32
    mixed_complex_dtype_testing: jnp.dtype = jnp.complex64

@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CCSDpTMeasCtx:
    rot_chol: jax.Array  # (n_chol, nocc, norb)
    lt1: jax.Array  # (n_chol, norb, nocc)
    cfg: CCSDpTMeasCfg  # static

    def tree_flatten(self):
        children = (self.rot_chol, self.lt1)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        rot_chol, lt1 = children
        return cls(rot_chol=rot_chol, lt1=lt1, cfg=cfg)

def build_meas_ctx(
    ham_data: HamChol, trial_data: CCSDpT_Trial, cfg: CCSDpTMeasCfg = CCSDpTMeasCfg()
) -> CCSDpTMeasCtx:
    chol = ham_data.chol  # (n_chol, norb, norb)
    nocc = trial_data.nocc
    rot_chol = chol[:, :nocc, :]  # (n_chol, nocc, norb)
    lt1 = jnp.einsum(
        "git,pt->gip",
        chol[:, :, nocc:],
        trial_data.t1,
        optimize="optimal",
        )  # (n_chol, norb, nocc)
    return CCSDpTMeasCtx(rot_chol=rot_chol, lt1=lt1, cfg=cfg)


def make_ccsd_pt_meas_ops(
    sys: System,
    memory_mode: str = "low",
    mixed_precision: bool = True,
    testing: bool = False,
) -> MeasOps:
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )

    cfg = CCSDpTMeasCfg(
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
    )

    return MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(
            ham_data, trial_data, cfg
        ),
        kernels={k_force_bias: force_bias_kernel_r, k_energy: energy_kernel_r},
    )