from trot import config

config.configure_once()

import functools

import jax
import jax.numpy as jnp
import pytest

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.meas.ucisdt import (
    build_meas_ctx,
    energy_kernel_rw_rh,
    energy_kernel_uw_rh,
    force_bias_kernel_rw_rh,
    force_bias_kernel_uw_rh,
    make_ucisdt_meas_ops,
)
from trot.trial.ucisdt import UcisdtTrial, make_ucisdt_trial_ops


def _completely_symmetrize_pairs_3(x: jax.Array) -> jax.Array:
    return (1.0 / 6.0) * (
        x
        + x.transpose(0, 1, 4, 5, 2, 3)
        + x.transpose(2, 3, 4, 5, 0, 1)
        + x.transpose(2, 3, 0, 1, 4, 5)
        + x.transpose(4, 5, 0, 1, 2, 3)
        + x.transpose(4, 5, 2, 3, 0, 1)
    )


def _completely_antisymmetrize_3(x: jax.Array) -> jax.Array:
    return (1.0 / 6.0) * (
        x
        - x.transpose(0, 1, 2, 5, 4, 3)
        + x.transpose(0, 3, 2, 5, 4, 1)
        - x.transpose(0, 3, 2, 1, 4, 5)
        + x.transpose(0, 5, 2, 1, 4, 3)
        - x.transpose(0, 5, 2, 3, 4, 1)
    )


def _enforce_particle_symmetries_triples_ccpy_like(
    c3aaa: jax.Array, c3aab: jax.Array, c3abb: jax.Array, c3bbb: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # UCISDT manual triples kernels are written for amplitudes in the same
    # symmetry class produced by ccpy_interface.prepare_ucc_amplitudes_from_ccpy.
    c3aaa = _completely_antisymmetrize_3(_completely_symmetrize_pairs_3(c3aaa))
    c3bbb = _completely_antisymmetrize_3(_completely_symmetrize_pairs_3(c3bbb))

    x3aab = 0.5 * (c3aab + c3aab.transpose(2, 3, 0, 1, 4, 5))
    x3aab = 0.5 * (x3aab - x3aab.transpose(0, 3, 2, 1, 4, 5))

    x3abb = 0.5 * (c3abb + c3abb.transpose(0, 1, 4, 5, 2, 3))
    x3abb = 0.5 * (x3abb - x3abb.transpose(0, 1, 2, 5, 4, 3))

    return c3aaa, x3aab, x3abb, c3bbb


def _make_ucisdt_trial(
    key,
    norb: int,
    nup: int,
    ndn: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
    scale_ci3: float = 0.01,
) -> UcisdtTrial:
    n_oa, n_ob = nup, ndn
    n_va = norb - n_oa
    n_vb = norb - n_ob
    keys = jax.random.split(key, 10)

    c1a = scale_ci1 * jax.random.normal(keys[0], (n_oa, n_va), dtype=dtype)
    c1b = scale_ci1 * jax.random.normal(keys[1], (n_ob, n_vb), dtype=dtype)
    c2aa = scale_ci2 * jax.random.normal(keys[2], (n_oa, n_va, n_oa, n_va), dtype=dtype)
    c2ab = scale_ci2 * jax.random.normal(keys[3], (n_oa, n_va, n_ob, n_vb), dtype=dtype)
    c2bb = scale_ci2 * jax.random.normal(keys[4], (n_ob, n_vb, n_ob, n_vb), dtype=dtype)
    c3aaa = scale_ci3 * jax.random.normal(keys[5], (n_oa, n_va, n_oa, n_va, n_oa, n_va), dtype=dtype)
    c3aab = scale_ci3 * jax.random.normal(keys[6], (n_oa, n_va, n_oa, n_va, n_ob, n_vb), dtype=dtype)
    c3abb = scale_ci3 * jax.random.normal(keys[7], (n_oa, n_va, n_ob, n_vb, n_ob, n_vb), dtype=dtype)
    c3bbb = scale_ci3 * jax.random.normal(keys[8], (n_ob, n_vb, n_ob, n_vb, n_ob, n_vb), dtype=dtype)

    # Antisymmetrize c2aa and c2bb under exchange of the two electron pairs
    c2aa = 0.25 * (
        c2aa
        - jnp.einsum("iajb->jaib", c2aa)
        - jnp.einsum("iajb->ibja", c2aa)
        + jnp.einsum("iajb->jbia", c2aa)
    )
    c2bb = 0.25 * (
        c2bb
        - jnp.einsum("iajb->jaib", c2bb)
        - jnp.einsum("iajb->ibja", c2bb)
        + jnp.einsum("iajb->jbia", c2bb)
    )
    c3aaa, c3aab, c3abb, c3bbb = _enforce_particle_symmetries_triples_ccpy_like(
        c3aaa, c3aab, c3abb, c3bbb
    )

    c_a = jnp.eye(norb, norb)
    c_b = testing.rand_orthonormal_cols(keys[9], norb, norb, dtype=jnp.float64)

    return UcisdtTrial(
        mo_coeff_a=c_a,
        mo_coeff_b=c_b,
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


# Full float64 precision for auto vs manual comparisons
_ucisdt_meas_ops_fp64 = functools.partial(
    make_ucisdt_meas_ops, mixed_precision=False, testing=True
)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 4, 2, 2, 5),
        ("unrestricted", 4, 2, 1, 5),
    ],
)
def test_auto_force_bias_matches_manual_ucisdt(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisdt_trial,
        make_trial_fn_kwargs=dict(norb=norb, nup=nup, ndn=ndn),
        make_trial_ops_fn=make_ucisdt_trial_ops,
        make_meas_ops_fn=_ucisdt_meas_ops_fp64,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(2):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, atol=1e-6), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 4, 2, 2, 5),
        ("unrestricted", 6, 3, 2, 8),
    ],
)
def test_auto_energy_matches_manual_ucisdt(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    key, k_w = jax.random.split(key)

    (
        sys,
        ham,
        trial,
        meas_manual,
        ctx_manual,
        meas_auto,
        ctx_auto,
    ) = testing.make_common_auto(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisdt_trial,
        make_trial_fn_kwargs=dict(norb=norb, nup=nup, ndn=ndn),
        make_trial_ops_fn=make_ucisdt_trial_ops,
        make_meas_ops_fn=_ucisdt_meas_ops_fp64,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(2):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        e_m = e_manual(wi, ham, ctx_manual, trial)
        e_a = e_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(e_a, e_m, rtol=5e-5, atol=5e-6), (e_a, e_m)


def test_force_bias_equal_when_wr_eq_wu():
    """Restricted and unrestricted force-bias kernels agree when walkers are equal."""
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    sys, ham, trial, ctx = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisdt_trial,
        make_trial_fn_kwargs=dict(norb=norb, nup=nup, ndn=ndn),
        make_trial_ops_fn=make_ucisdt_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wa, wb = wi
        wi = (wa, wa)
        fbu = force_bias_kernel_uw_rh(wi, ham, ctx, trial)
        fbr = force_bias_kernel_rw_rh(wa, ham, ctx, trial)
        assert jnp.allclose(fbu, fbr, atol=1e-12), (fbu, fbr)


def test_energy_equal_when_wr_eq_wu():
    """Restricted and unrestricted energy kernels agree when walkers are equal."""
    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    walker_kind = "unrestricted"

    key = jax.random.PRNGKey(1)
    key, k_w = jax.random.split(key)

    sys, ham, trial, ctx = testing.make_common_manual_only(
        key,
        walker_kind,
        norb,
        (nup, ndn),
        n_chol,
        make_trial_fn=_make_ucisdt_trial,
        make_trial_fn_kwargs=dict(norb=norb, nup=nup, ndn=ndn),
        make_trial_ops_fn=make_ucisdt_trial_ops,
        build_meas_ctx_fn=build_meas_ctx,
    )

    for i in range(4):
        wi = testing.make_walkers(jax.random.fold_in(k_w, i), sys)
        wa, wb = wi
        wi = (wa, wa)
        eu = energy_kernel_uw_rh(wi, ham, ctx, trial)
        er = energy_kernel_rw_rh(wa, ham, ctx, trial)
        assert jnp.allclose(eu, er, atol=1e-12), (eu, er)


def test_generalized_walkers_raise():
    from trot.core.system import System
    sys = System(norb=4, nelec=(2, 2), walker_kind="generalized")
    with pytest.raises(NotImplementedError):
        make_ucisdt_meas_ops(sys)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
    ],
)
def test_low_memory_matches_high_memory_ucisdt(walker_kind, norb, nup, ndn, n_chol):
    from trot.core.system import System

    key = jax.random.PRNGKey(11)
    key, k_ham, k_trial, k_w = jax.random.split(key, 4)

    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)
    ham = testing.make_random_ham_chol(
        k_ham, norb=norb, n_chol=n_chol, basis="restricted", dtype=jnp.float64
    )
    trial = _make_ucisdt_trial(k_trial, norb=norb, nup=nup, ndn=ndn, dtype=jnp.float64)

    meas_high = make_ucisdt_meas_ops(sys, memory_mode="high", mixed_precision=False, testing=True)
    meas_low = make_ucisdt_meas_ops(sys, memory_mode="low", mixed_precision=False, testing=True)

    ctx_high = meas_high.build_meas_ctx(ham, trial)
    ctx_low = meas_low.build_meas_ctx(ham, trial)

    fb_high = meas_high.require_kernel(k_force_bias)
    fb_low = meas_low.require_kernel(k_force_bias)
    e_high = meas_high.require_kernel(k_energy)
    e_low = meas_low.require_kernel(k_energy)

    for i in range(2):
        walker = testing.make_walkers(jax.random.fold_in(k_w, i), sys, dtype=jnp.complex128)
        fb_h = fb_high(walker, ham, ctx_high, trial)
        fb_l = fb_low(walker, ham, ctx_low, trial)
        e_h = e_high(walker, ham, ctx_high, trial)
        e_l = e_low(walker, ham, ctx_low, trial)

        assert jnp.allclose(fb_l, fb_h, rtol=1e-11, atol=1e-11), (fb_l, fb_h)
        assert jnp.allclose(e_l, e_h, rtol=1e-11, atol=1e-11), (e_l, e_h)


def test_memory_mode_defaults_to_high_and_allows_low_override():
    from trot.core.system import System

    norb = 6
    nup, ndn = 2, 2
    n_chol = 8
    key = jax.random.PRNGKey(5)
    key, k_ham, k_trial = jax.random.split(key, 3)

    sys = System(norb=norb, nelec=(nup, ndn), walker_kind="unrestricted")
    ham = testing.make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol, basis="restricted")
    trial = _make_ucisdt_trial(k_trial, norb=norb, nup=nup, ndn=ndn)

    ctx_default = make_ucisdt_meas_ops(sys).build_meas_ctx(ham, trial)
    ctx_low = make_ucisdt_meas_ops(sys, memory_mode="low").build_meas_ctx(ham, trial)

    assert ctx_default.cfg.memory_mode == "high"
    assert ctx_low.cfg.memory_mode == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
