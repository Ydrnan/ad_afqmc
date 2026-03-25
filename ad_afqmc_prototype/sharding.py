from typing import TypeVar, cast

import jax
from jax import tree_util
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .ham.chol import HamChol
from .ham.hubbard import HamHubbard
from .prop.types import PropState
from .trial.ghf import GhfTrial
from .trial.rhf import RhfTrial
from .trial.uhf import UhfTrial

THam = TypeVar("THam")
TTrial = TypeVar("TTrial")


def make_data_mesh() -> Mesh:
    n = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n,))
    return Mesh(devices, ("data",))


def make_data_model_mesh(n_data: int | None = None, n_model: int | None = None) -> Mesh:
    n = jax.local_device_count()

    if n_data is None and n_model is None:
        n_data, n_model = 1, n
    elif n_data is None:
        assert n_model is not None
        if n % n_model != 0:
            raise ValueError(f"local_device_count={n} is not divisible by n_model={n_model}.")
        n_data = n // n_model
    elif n_model is None:
        if n % n_data != 0:
            raise ValueError(f"local_device_count={n} is not divisible by n_data={n_data}.")
        n_model = n // n_data

    assert n_data is not None and n_model is not None
    if n_data * n_model != n:
        raise ValueError(
            f"Requested mesh ({n_data}, {n_model}) uses {n_data * n_model} devices, "
            f"but {n} local devices are visible."
        )

    devices = mesh_utils.create_device_mesh((n_data, n_model))
    return Mesh(devices, ("data", "model"))


def has_model_axis(mesh: Mesh | None) -> bool:
    return mesh is not None and "model" in mesh.axis_names


def shard_first_axis(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def shard_model_axis(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P("model")))


def replicate(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P()))


def shard_ham_data(ham_data: THam, mesh: Mesh | None) -> THam:
    """
    For a data x model mesh:
      - replicate h0/h1
      - shard chol on the model axis
    """
    if mesh is None or mesh.size == 1 or not has_model_axis(mesh):
        return ham_data

    if isinstance(ham_data, HamChol):
        return cast(
            THam,
            HamChol(
                h0=replicate(ham_data.h0, mesh),
                h1=replicate(ham_data.h1, mesh),
                chol=shard_model_axis(ham_data.chol, mesh),
                basis=ham_data.basis,
            ),
        )

    if isinstance(ham_data, HamHubbard):
        raise ValueError("Cannot shard Hubbard Hamiltonian, don't use model axis sharding.")

    return ham_data


def shard_hf_trial_data(trial_data: TTrial, mesh: Mesh | None) -> TTrial:
    """
    For HF like trial objects we keep MO coefficients replicated.

    Unsupported trial data is left unchanged for now.
    """
    if mesh is None or mesh.size == 1 or not has_model_axis(mesh):
        return trial_data

    if isinstance(trial_data, RhfTrial):
        return cast(TTrial, RhfTrial(mo_coeff=replicate(trial_data.mo_coeff, mesh)))

    if isinstance(trial_data, UhfTrial):
        return cast(
            TTrial,
            UhfTrial(
                mo_coeff_a=replicate(trial_data.mo_coeff_a, mesh),
                mo_coeff_b=replicate(trial_data.mo_coeff_b, mesh),
            ),
        )

    if isinstance(trial_data, GhfTrial):
        return cast(TTrial, GhfTrial(mo_coeff=replicate(trial_data.mo_coeff, mesh)))

    return trial_data


def shard_runtime_inputs(
    ham_data: THam, trial_data: TTrial, mesh: Mesh | None
) -> tuple[THam, TTrial]:
    return shard_ham_data(ham_data, mesh), shard_hf_trial_data(trial_data, mesh)


def shard_prop_state(state: PropState, mesh: Mesh | None) -> PropState:
    """
    Shard only (n_walkers,...) leaves, keep global scalars replicated.
    """
    if mesh is None or mesh.size == 1:
        return state

    walkers_sh = tree_util.tree_map(lambda a: shard_first_axis(a, mesh), state.walkers)

    return state._replace(
        walkers=walkers_sh,
        weights=shard_first_axis(state.weights, mesh),
        overlaps=shard_first_axis(state.overlaps, mesh),
        rng_key=replicate(state.rng_key, mesh),
        pop_control_ene_shift=replicate(state.pop_control_ene_shift, mesh),
        e_estimate=replicate(state.e_estimate, mesh),
        node_encounters=replicate(state.node_encounters, mesh),
    )
