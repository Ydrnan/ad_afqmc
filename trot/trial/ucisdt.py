from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdtTrial:
    """
    Unrestricted CISDT trial in an MO basis where the reference
    determinant occupies the first nocc[0] alpha and nocc[1] beta orbitals.

    Arrays:
      mo_coeff_a: (norb, norb)
      mo_coeff_b: (norb, norb)                                  beta MO rotation matrix
      c1a : (nocc[0], nvir[0])                                  alpha singles
      c1b : (nocc[1], nvir[1])                                  beta singles
      c2aa: (nocc[0], nvir[0], nocc[0], nvir[0])               alpha-alpha doubles
      c2ab: (nocc[0], nvir[0], nocc[1], nvir[1])               alpha-beta doubles
      c2bb: (nocc[1], nvir[1], nocc[1], nvir[1])               beta-beta doubles
      c3aaa: (nocc[0], nvir[0], nocc[0], nvir[0], nocc[0], nvir[0])  AAA triples
      c3aab: (nocc[0], nvir[0], nocc[0], nvir[0], nocc[1], nvir[1])  AAB triples
      c3abb: (nocc[0], nvir[0], nocc[1], nvir[1], nocc[1], nvir[1])  ABB triples
      c3bbb: (nocc[1], nvir[1], nocc[1], nvir[1], nocc[1], nvir[1])  BBB triples
    """

    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    c2aa: jax.Array
    c2ab: jax.Array
    c2bb: jax.Array
    c3aaa: jax.Array
    c3aab: jax.Array
    c3abb: jax.Array
    c3bbb: jax.Array

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_b.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (int(self.c1a.shape[0]), int(self.c1b.shape[0]))

    @property
    def nvir(self) -> tuple[int, int]:
        return (int(self.c1a.shape[1]), int(self.c1b.shape[1]))

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.c2aa,
            self.c2ab,
            self.c2bb,
            self.c3aaa,
            self.c3aab,
            self.c3abb,
            self.c3bbb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            mo_coeff_a,
            mo_coeff_b,
            c1a,
            c1b,
            c2aa,
            c2ab,
            c2bb,
            c3aaa,
            c3aab,
            c3abb,
            c3bbb,
        ) = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b,
            c1a=c1a,
            c1b=c1b,
            c2aa=c2aa,
            c2ab=c2ab,
            c2bb=c2bb,
            c3aaa=c3aaa,
            c3aab=c3aab,
            c3abb=c3abb,
            c3bbb=c3bbb,
        )


def get_rdm1(trial_data: UcisdtTrial) -> jax.Array:
    norb, (n_oa, n_ob) = trial_data.norb, trial_data.nocc
    occ_a = jnp.arange(norb) < n_oa
    c_b = trial_data.mo_coeff_b
    dm_a = jnp.diag(occ_a)
    dm_b = c_b[:, :n_ob] @ c_b[:, :n_ob].conj().T
    return jnp.stack([dm_a, dm_b], axis=0)


def overlap_r(walker: jax.Array, trial_data: UcisdtTrial) -> jax.Array:
    n_elec_0 = trial_data.nocc[0]
    n_elec_1 = trial_data.nocc[1]
    return overlap_u((walker[:, :n_elec_0], walker[:, :n_elec_1]), trial_data)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UcisdtTrial) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c_b = trial_data.mo_coeff_b

    wb = c_b.T @ wb[:, :n_ob]
    woa = wa[:n_oa, :]
    wob = wb[:n_ob, :]

    g_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    g_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    g_a_vir = g_a[:, n_oa:]
    g_b_vir = g_b[:, n_ob:]

    o0 = jnp.linalg.det(woa) * jnp.linalg.det(wob)
    o1 = jnp.einsum("ia,ia", trial_data.c1a, g_a_vir) + jnp.einsum(
        "ia,ia", trial_data.c1b, g_b_vir
    )
    o2 = (
        0.5 * jnp.einsum("iajb,ia,jb", trial_data.c2aa, g_a_vir, g_a_vir)
        + 0.5 * jnp.einsum("iajb,ia,jb", trial_data.c2bb, g_b_vir, g_b_vir)
        + jnp.einsum("iajb,ia,jb", trial_data.c2ab, g_a_vir, g_b_vir)
    )
    o3 = (
        (1 / 6) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3aaa, g_a_vir, g_a_vir, g_a_vir)
        + (1 / 6) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3bbb, g_b_vir, g_b_vir, g_b_vir)
        + (1 / 2) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3aab, g_a_vir, g_a_vir, g_b_vir)
        + (1 / 2) * jnp.einsum("iajbkc,ia,jb,kc", trial_data.c3abb, g_a_vir, g_b_vir, g_b_vir)
    )
    return (1.0 + o1 + o2 + o3) * o0


def make_ucisdt_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)
    if wk == "generalized":
        raise NotImplementedError("UCISDT does not support generalized walkers.")
    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


def make_ucisdt_trial_data(data: dict, sys: System) -> UcisdtTrial:
    return UcisdtTrial(
        mo_coeff_a=jnp.asarray(data["mo_coeff_a"]),
        mo_coeff_b=jnp.asarray(data["mo_coeff_b"]),
        c1a=jnp.asarray(data["ci1a"]),
        c1b=jnp.asarray(data["ci1b"]),
        c2aa=jnp.asarray(data["ci2aa"]),
        c2ab=jnp.asarray(data["ci2ab"]),
        c2bb=jnp.asarray(data["ci2bb"]),
        c3aaa=jnp.asarray(data["ci3aaa"]),
        c3aab=jnp.asarray(data["ci3aab"]),
        c3abb=jnp.asarray(data["ci3abb"]),
        c3bbb=jnp.asarray(data["ci3bbb"]),
    )
