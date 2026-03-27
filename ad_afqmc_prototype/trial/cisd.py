from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdTrial:
    """
    Restricted CISD trial in an MO basis where the reference
    determinant occupies the first ``nocc_full`` orbitals.

    Arrays:
      ci1: (nocc_act, nvir_act)                 singles coefficients c_{i a}
      ci2: (nocc_act, nvir_act, nocc_act, nvir_act)
           doubles coefficients c_{i a j b}

    Layout in the full AFQMC correlation space:
      [trial-core | trial-active-occ | trial-active-vir | trial-outer]

    The CI amplitudes refer only to the active occupied/virtual blocks.
    """

    ci1: jax.Array
    ci2: jax.Array
    nocc_t_core: int = 0
    nvir_t_outer: int = 0

    @property
    def nocc(self) -> int:
        """Number of active occupied orbitals."""
        return int(self.ci1.shape[0])

    @property
    def nvir(self) -> int:
        """Number of active virtual orbitals."""
        return int(self.ci1.shape[1])

    @property
    def nocc_full(self) -> int:
        """Number of occupied orbitals in the full AFQMC correlation space."""
        return int(self.nocc_t_core + self.nocc)

    @property
    def nvir_full(self) -> int:
        """Number of virtual orbitals in the full AFQMC correlation space."""
        return int(self.nvir + self.nvir_t_outer)

    @property
    def norb(self) -> int:
        """Number of orbitals in the full AFQMC correlation space."""
        return int(self.nocc_full + self.nvir_full)

    @property
    def occ_act_slice(self) -> slice:
        return slice(self.nocc_t_core, self.nocc_full)

    @property
    def vir_act_slice(self) -> slice:
        return slice(self.nocc_full, self.nocc_full + self.nvir)

    @property
    def norb_act(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        children = (self.ci1, self.ci2)
        aux = (self.nocc_t_core, self.nvir_t_outer)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        nocc_t_core, nvir_t_outer = aux
        ci1, ci2 = children
        return cls(
            ci1=ci1,
            ci2=ci2,
            nocc_t_core=nocc_t_core,
            nvir_t_outer=nvir_t_outer,
        )


def get_rdm1(trial_data: CisdTrial) -> jax.Array:
    # RHF
    norb, nocc = trial_data.norb, trial_data.nocc_full
    occ = jnp.arange(norb) < nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def overlap_r(walker: jax.Array, trial_data: CisdTrial) -> jax.Array:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc_full = trial_data.nocc_full

    wocc = walker[:nocc_full, :]  # (nocc_full, nocc_full)
    green = jnp.linalg.solve(wocc.T, walker.T)  # (nocc, norb)

    det0 = jnp.linalg.det(wocc)
    o0 = det0 * det0

    x = green[trial_data.occ_act_slice, trial_data.vir_act_slice]  # (nocc_act, nvir_act)
    o1 = jnp.einsum("ia,ia->", ci1, x)
    o2 = 2.0 * jnp.einsum("iajb,ia,jb->", ci2, x, x) - jnp.einsum("iajb,ib,ja->", ci2, x, x)

    return (1.0 + 2.0 * o1 + o2) * o0


def make_cisd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("Restricted CISD trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)


def make_cisd_trial_data(data: dict, sys: System) -> CisdTrial:
    ci1 = jnp.asarray(data["ci1"])
    ci2 = jnp.asarray(data["ci2"])
    nocc_t_core = int(jnp.asarray(data.get("nocc_t_core", 0)).item())
    nvir_t_outer = int(jnp.asarray(data.get("nvir_t_outer", 0)).item())
    return CisdTrial(
        ci1=ci1,
        ci2=ci2,
        nocc_t_core=nocc_t_core,
        nvir_t_outer=nvir_t_outer,
    )


def slice_trial_level(trial: CisdTrial, nvir_keep: int | None) -> CisdTrial:
    """
    Return a trial object whose ci1/ci2 are sliced to keep only the first nvir_keep virtuals.
    """
    if nvir_keep is None:
        return trial

    ci1 = trial.ci1[:, :nvir_keep]
    ci2 = trial.ci2[:, :nvir_keep, :, :nvir_keep]
    return replace(
        trial,
        ci1=ci1,
        ci2=ci2,
        nvir_t_outer=trial.nvir_t_outer + (trial.nvir - nvir_keep),
    )
