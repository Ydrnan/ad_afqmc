import os
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import ad_afqmc_prototype.walkers as wk
from ad_afqmc_prototype import testing
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.prop.afqmc import init_prop_state, make_prop_ops
from ad_afqmc_prototype.prop.blocks import block
from ad_afqmc_prototype.prop.chol_afqmc_ops import _build_prop_ctx
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.sharding import make_data_mesh, shard_prop_state


@pytest.mark.parametrize("n_per_dev", [4])
def test_block_runs_under_sharding(n_per_dev):

    mesh = make_data_mesh()
    ndev = mesh.size

    norb, nocc, n_chol = 6, 3, 4
    nw = ndev * n_per_dev

    rng = jax.random.PRNGKey(42)
    k_ham, k_walk, k_walk2, k_wt = jax.random.split(rng, 4)

    ham = testing.make_random_ham_chol(k_ham, norb, n_chol)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")

    trial_ops = testing.make_dummy_trial_ops()
    meas_ops = make_auto_meas_ops(sys, trial_ops)
    trial_data = {"rdm1": jnp.eye(norb, dtype=jnp.float64)}
    meas_ctx = meas_ops.build_meas_ctx(ham, trial_data)

    params = QmcParams(
        dt=0.1,
        n_chunks=1,
        n_exp_terms=4,
        n_prop_steps=1,
        shift_ema=0.5,
        n_walkers=nw,
        seed=0,
        pop_control_damping=0.1,
    )

    prop_ops = make_prop_ops(ham.basis, sys.walker_kind)
    prop_ctx = _build_prop_ctx(ham, trial_data["rdm1"], params.dt)

    initial_walkers = jax.random.normal(
        k_walk, (nw, norb, nocc), dtype=jnp.float64
    ) + 1.0j * jax.random.normal(k_walk2, (nw, norb, nocc), dtype=jnp.float64)

    # unsharded reference
    state0_u = init_prop_state(
        sys=sys,
        ham_data=ham,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        params=params,
        initial_walkers=initial_walkers,
        mesh=None,
    )
    rand_weights = jax.random.uniform(k_wt, (nw,), dtype=jnp.float64, minval=0.5, maxval=2.0)
    state0_u = state0_u._replace(weights=rand_weights)

    # sharded version of the exact same state
    state0_s = shard_prop_state(state0_u, mesh)

    def _assert_sharded_first_axis(a):
        assert isinstance(a.sharding, NamedSharding)
        assert a.sharding.spec == P("data")

    def _assert_replicated(a):
        assert isinstance(a.sharding, NamedSharding)
        assert a.sharding.spec == P()

    for leaf in tree_util.tree_leaves(state0_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state0_s.weights)
    _assert_sharded_first_axis(state0_s.overlaps)
    _assert_replicated(state0_s.rng_key)
    _assert_replicated(state0_s.e_estimate)
    _assert_replicated(state0_s.pop_control_ene_shift)

    wsum_direct = jnp.sum(state0_s.weights)
    wsum_expected = float(jax.device_get(jnp.sum(rand_weights)))
    assert float(jax.device_get(wsum_direct)) == pytest.approx(wsum_expected, abs=1e-12)

    data_sh = NamedSharding(mesh, P("data"))
    sr_sharded = partial(wk.stochastic_reconfiguration, data_sharding=data_sh)
    sr_unsharded = partial(wk.stochastic_reconfiguration, data_sharding=None)

    def _call_block(st, *, sr_fn):
        return block(
            st,
            sys=sys,
            params=params,
            ham_data=ham,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            sr_fn=sr_fn,
        )

    run_block_u = jax.jit(partial(_call_block, sr_fn=sr_unsharded))
    run_block_s = jax.jit(partial(_call_block, sr_fn=sr_sharded))

    state1_u, obs_u = run_block_u(state0_u)
    state1_s, obs_s = run_block_s(state0_s)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["weight"])),
        np.asarray(jax.device_get(obs_u.scalars["weight"])),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(obs_s.scalars["energy"])),
        np.asarray(jax.device_get(obs_u.scalars["energy"])),
        rtol=1e-12,
        atol=1e-12,
    )

    # walkers and weights must match after SR
    for leaf_s, leaf_u in zip(
        tree_util.tree_leaves(state1_s.walkers),
        tree_util.tree_leaves(state1_u.walkers),
    ):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(leaf_s)),
            np.asarray(jax.device_get(leaf_u)),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(state1_s.weights)),
        np.asarray(jax.device_get(state1_u.weights)),
        rtol=1e-12,
        atol=1e-12,
    )

    for leaf in tree_util.tree_leaves(state1_s.walkers):
        _assert_sharded_first_axis(leaf)
    _assert_sharded_first_axis(state1_s.weights)
    _assert_sharded_first_axis(state1_s.overlaps)
    _assert_replicated(state1_s.rng_key)
    _assert_replicated(state1_s.e_estimate)
    _assert_replicated(state1_s.pop_control_ene_shift)


@pytest.mark.parametrize("n_per_dev", [4])
def test_sr_sharded_matches_unsharded(n_per_dev):
    """Stochastic reconfiguration gives identical results with and without sharding."""

    mesh = make_data_mesh()
    ndev = mesh.size
    nw = ndev * n_per_dev
    norb, nocc = 6, 3

    rng = jax.random.PRNGKey(99)
    k1, k2, k3 = jax.random.split(rng, 3)
    walkers = jax.random.normal(k1, (nw, norb, nocc), dtype=jnp.float64) + 1.0j * jax.random.normal(
        k2, (nw, norb, nocc), dtype=jnp.float64
    )
    weights = jax.random.uniform(k3, (nw,), dtype=jnp.float64, minval=0.5, maxval=2.0)
    zeta = jax.random.uniform(jax.random.PRNGKey(7), (), dtype=jnp.float64)

    data_sh = NamedSharding(mesh, P("data"))
    walkers_s = jax.device_put(walkers, data_sh)
    weights_s = jax.device_put(weights, data_sh)

    sr = jax.jit(wk.stochastic_reconfiguration, static_argnames=("walker_kind", "data_sharding"))

    w_u, wt_u = sr(walkers, weights, zeta, "restricted", data_sharding=None)
    w_s, wt_s = sr(walkers_s, weights_s, zeta, "restricted", data_sharding=data_sh)

    np.testing.assert_allclose(
        np.asarray(jax.device_get(w_s)),
        np.asarray(jax.device_get(w_u)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(wt_s)),
        np.asarray(jax.device_get(wt_u)),
        rtol=1e-12,
        atol=1e-12,
    )

    # output should retain sharding
    assert isinstance(w_s.sharding, NamedSharding)
    assert w_s.sharding.spec == P("data")
    assert isinstance(wt_s.sharding, NamedSharding)
    assert wt_s.sharding.spec == P("data")


if __name__ == "__main__":
    test_sr_sharded_matches_unsharded(n_per_dev=4)
    # test_block_runs_under_sharding(n_per_dev=4)
