import pytest
import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.spin_proj
import trot.testing

import jax
import jax.numpy as jnp

mol = gto.M(
    atom="""
    N                 -1.67119571   -1.44021737    0.00000000
    H                 -2.12619571   -0.65213425    1.00000000
    H                 -0.76119571   -1.44021737    1.00000000
    """,
    basis="6-31g",
    spin=1,
)
mf = scf.UHF(mol)
mf.kernel()

mycc = cc.UCCSD(mf)
mycc.kernel()

af = AfqmcFp(mycc)
af.dt = 0.1
af.n_walkers = 10
af.ene0 = mycc.e_tot
af.seed = 5
af.n_prop_steps = 50
af.n_blocks = 1
af.walker_kind = "unrestricted"
af.n_traj = 10
af.mixed_precision = False
af.build_job()
job = af._job


@pytest.mark.parametrize(
    "target_spin, e_ref, err_ref",
    [
        (1.0, -55.559999973012, 1.5120309e-03),
    ],
)
def test_spin_proj_s2(target_spin, e_ref, err_ref):
    from trot.meas.ucisd import energy_kernel_gw_rh
    from trot.trial.ucisd import overlap_g
    from trot.core.ops import k_energy
    from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2

    # Spin projection
    ## Data for the quadrature
    betas, w_betas = trot.spin_proj.quadrature_s2(
        target_spin,
        (job.sys.nup, job.sys.ndn),
        ngrid=4,
    )

    ## Overlap and energy with spin projection
    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
        betas, w_betas, overlap_g, energy_kernel_gw_rh
    )

    job.meas_ops = dataclasses.replace(
        job.meas_ops,
        overlap=overlap_u_s2,
        kernels={
            k_energy: energy_kernel_uw_rh_s2,
        },
    )

    # Important to do after changing the energy and overlap kernels if the
    # function job._prepare_runtime() has been run before. Otherwise the initial
    # state will not be the right one.
    job._runtime_prop_ctx = None
    job._runtime_meas_ctx = None
    job._runtime_state = None
    job._prepare_runtime()

    e, err = af.kernel()

    assert abs(e[-1].real - e_ref) < 1e-6, (e[-1].real, e_ref)
    assert abs(err[-1].real - err_ref) < 1e-6, (err[-1].real, err_ref)


@pytest.mark.parametrize(
    "target_spin",
    [
        (1),
        (3),
        (5),
    ],
)
def test_quadrature(target_spin):
    key = jax.random.key(42)
    wa, wb = trot.testing.make_walkers(key, job.sys)
    w = (wa, wb)

    from trot.meas.ucisd import energy_kernel_gw_rh
    from trot.trial.ucisd import overlap_g
    from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2

    # Spin projection
    ## Data for the quadrature
    betas, w_betas = trot.spin_proj.quadrature_s2(
        target_spin,
        (job.sys.nup, job.sys.ndn),
        ngrid=10,
    )

    ## Overlap and energy with spin projection
    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
        betas, w_betas, overlap_g, energy_kernel_gw_rh
    )

    trial_data = job.trial_data
    meas_ctx = job._runtime_meas_ctx
    ham_data = job.ham_data

    o1 = overlap_u_s2(w, trial_data)
    e1 = energy_kernel_uw_rh_s2(w, ham_data, meas_ctx, trial_data)

    # Spin projection
    ## Brute force
    S = target_spin / 2.0
    Sz = (job.sys.nup - job.sys.ndn) / 2.0

    ngrid = 1000
    betas = jnp.linspace(0, jnp.pi, ngrid, endpoint=False)
    wigner = lambda beta: trot.spin_proj.wigner_small_d(S, Sz, Sz, beta)
    w_betas = jax.vmap(wigner)(betas) * jnp.sin(betas) * (2 * S + 1) / 2.0 * jnp.pi / ngrid

    ## Overlap and energy with spin projection
    overlap_u_s2 = make_overlap_u_s2(betas, w_betas, overlap_g)
    energy_kernel_uw_rh_s2 = make_energy_kernel_uw_rh_s2(
        betas, w_betas, overlap_g, energy_kernel_gw_rh
    )

    o2 = overlap_u_s2(w, trial_data)
    e2 = energy_kernel_uw_rh_s2(w, ham_data, meas_ctx, trial_data)

    assert jnp.isclose(o1, o2), (o1, o2)
    assert jnp.isclose(e1.real, e2.real), (e1, e2)


if __name__ == "__main__":
    pytest.main([__file__])
