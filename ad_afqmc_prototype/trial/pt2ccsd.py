from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2CCSDTrial:
    """
    Restricted pt2CCSD trial in an MO basis where the reference
    determinant occupies the first nocc orbitals.

    Arrays:
      t1: (nocc, nvir)                     singles coefficients t_{i a}
      t2: (nocc, nvir, nocc, nvir)         doubles coefficients t_{i a j b}
    """

    t1: jax.Array
    t2: jax.Array

    @property
    def nocc(self) -> int:
        return int(self.t1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.t1.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        children = (self.t1, self.t2)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (t1, t2) = children
        return cls(
            t1=t1,
            t2=t2,
        )


def get_rdm1(trial_data: Pt2CCSDTrial) -> jax.Array:
    # RHF
    norb, nocc = trial_data.norb, trial_data.nocc
    occ = jnp.arange(norb) < nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


# def _det(m: jax.Array) -> jax.Array:
#     return jnp.linalg.det(m)


def overlap_r(walker: jax.Array, trial_data: Pt2CCSDTrial) -> jax.Array:
    # <HF|walker>
    nocc = trial_data.nocc
    wocc = walker[:nocc, :]  # (nocc, nocc)
    det0 = jnp.linalg.det(wocc)
    o0 = det0 * det0
    return o0


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: Pt2CCSDTrial) -> jax.Array:
    wu, wd = walker
    nocc = trial_data.nocc
    wu_occ = wu[:nocc, :]
    wd_occ = wd[:nocc, :]
    detu = jnp.linalg.det(wu_occ)  # (nocc_a, nocc_a)
    detd = jnp.linalg.det(wd_occ)  # (nocc_b, nocc_b)
    return detu * detd


# def overlap_g(walker: jax.Array, trial_data: RhfTrial) -> jax.Array:
#     norb = trial_data.norb
#     cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
#     top = cH @ walker[:norb, :]  # (nocc, 2*nocc)
#     bot = cH @ walker[norb:, :]  # (nocc, 2*nocc)
#     m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)
#     return _det(m)


def make_ccsd_pt_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("RHF requires nelec[0] == nelec[1].")

    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
    
    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)

    # if wk == "generalized":
    #     return TrialOps(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
