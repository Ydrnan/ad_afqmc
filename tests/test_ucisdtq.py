from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import pytest

from trot import testing
from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas.auto import make_auto_meas_ops
from trot.trial.ucisdtq import UcisdtqTrial, overlap_r, overlap_u, make_ucisdtq_trial_ops


def _make_ucisdtq_trial(
    key,
    norb: int,
    nup: int,
    ndn: int,
    *,
    dtype=jnp.float64,
    scale_ci1: float = 0.05,
    scale_ci2: float = 0.02,
    scale_ci3: float = 0.005,
    scale_ci4: float = 0.002,
) -> UcisdtqTrial:
    n_oa, n_ob = nup, ndn
    n_va = norb - n_oa
    n_vb = norb - n_ob
    keys = jax.random.split(key, 16)

    c1a = scale_ci1 * jax.random.normal(keys[0], (n_oa, n_va), dtype=dtype)
    c1b = scale_ci1 * jax.random.normal(keys[1], (n_ob, n_vb), dtype=dtype)
    c2aa = scale_ci2 * jax.random.normal(keys[2], (n_oa, n_va, n_oa, n_va), dtype=dtype)
    c2ab = scale_ci2 * jax.random.normal(keys[3], (n_oa, n_va, n_ob, n_vb), dtype=dtype)
    c2bb = scale_ci2 * jax.random.normal(keys[4], (n_ob, n_vb, n_ob, n_vb), dtype=dtype)
    c3aaa = scale_ci3 * jax.random.normal(
        keys[5], (n_oa, n_va, n_oa, n_va, n_oa, n_va), dtype=dtype
    )
    c3aab = scale_ci3 * jax.random.normal(
        keys[6], (n_oa, n_va, n_oa, n_va, n_ob, n_vb), dtype=dtype
    )
    c3abb = scale_ci3 * jax.random.normal(
        keys[7], (n_oa, n_va, n_ob, n_vb, n_ob, n_vb), dtype=dtype
    )
    c3bbb = scale_ci3 * jax.random.normal(
        keys[8], (n_ob, n_vb, n_ob, n_vb, n_ob, n_vb), dtype=dtype
    )
    c4aaaa = scale_ci4 * jax.random.normal(
        keys[9], (n_oa, n_va, n_oa, n_va, n_oa, n_va, n_oa, n_va), dtype=dtype
    )
    c4aaab = scale_ci4 * jax.random.normal(
        keys[10], (n_oa, n_va, n_oa, n_va, n_oa, n_va, n_ob, n_vb), dtype=dtype
    )
    c4aabb = scale_ci4 * jax.random.normal(
        keys[11], (n_oa, n_va, n_oa, n_va, n_ob, n_vb, n_ob, n_vb), dtype=dtype
    )
    c4abbb = scale_ci4 * jax.random.normal(
        keys[12], (n_oa, n_va, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb), dtype=dtype
    )
    c4bbbb = scale_ci4 * jax.random.normal(
        keys[13], (n_ob, n_vb, n_ob, n_vb, n_ob, n_vb, n_ob, n_vb), dtype=dtype
    )

    c_a = jnp.eye(norb, norb)
    c_b = testing.rand_orthonormal_cols(keys[14], norb, norb, dtype=jnp.float64)

    return UcisdtqTrial(
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
        c4aaaa=c4aaaa,
        c4aaab=c4aaab,
        c4aabb=c4aabb,
        c4abbb=c4abbb,
        c4bbbb=c4bbbb,
    )


@pytest.mark.parametrize(
    "norb,nup,ndn",
    [
        (4, 2, 2),
        (4, 2, 1),
    ],
)
def test_overlap_r_matches_u_when_equal(norb, nup, ndn):
    """overlap_r and overlap_u agree when alpha and beta share the same orbital columns."""
    key = jax.random.PRNGKey(7)
    k_trial, k_w = jax.random.split(key)

    trial = _make_ucisdtq_trial(k_trial, norb, nup, ndn)

    for i in range(3):
        # overlap_r(wa, trial) internally calls overlap_u((wa[:, :nup], wa[:, :ndn]), trial)
        wa = testing.rand_orthonormal_cols(
            jax.random.fold_in(k_w, i), norb, nup, dtype=jnp.complex128
        )
        ov_r = overlap_r(wa, trial)
        ov_u = overlap_u((wa, wa[:, :ndn]), trial)
        assert jnp.allclose(ov_r, ov_u, atol=1e-12), (ov_r, ov_u)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 4, 2, 2, 5),
        ("unrestricted", 4, 2, 1, 5),
    ],
)
def test_auto_meas_runs_ucisdtq(walker_kind, norb, nup, ndn, n_chol):
    """Auto measurement ops compile and produce finite outputs for UCISDTQ."""
    key = jax.random.PRNGKey(3)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)
    ham = testing.make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_ucisdtq_trial(k_trial, norb, nup, ndn)
    t_ops = make_ucisdtq_trial_ops(sys)
    meas = make_auto_meas_ops(sys, t_ops)
    ctx = meas.build_meas_ctx(ham, trial)

    fb = meas.require_kernel(k_force_bias)
    e = meas.require_kernel(k_energy)

    wi = testing.make_walkers(k_w, sys)
    fb_val = fb(wi, ham, ctx, trial)
    e_val = e(wi, ham, ctx, trial)

    assert fb_val.shape == (n_chol,)
    assert e_val.shape == ()
    assert jnp.isfinite(fb_val).all()
    assert jnp.isfinite(e_val)


def test_generalized_walkers_raise():
    sys = System(norb=4, nelec=(2, 2), walker_kind="generalized")
    with pytest.raises(NotImplementedError):
        make_ucisdtq_trial_ops(sys)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
