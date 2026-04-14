import dataclasses
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcFp
import trot.trial.uccsd
import trot.spin_proj

import jax.numpy as jnp

e_ref = -108.8635909545
err_ref = 3.8744131e-03


def test_spin_proj_s2():
    mol = gto.M(
        atom="""
        N 0.0 0.0 0.0
        N 0.0 0.0 2.0
        """,
        basis="6-31g",
        verbose=3,
    )

    mf = scf.UHF(mol)
    mf.kernel()

    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability()

    mycc = cc.UCCSD(mf)
    mycc.kernel()

    af = AfqmcFp(mycc)
    af.n_walkers = 10
    af.ene0 = mycc.e_tot
    af.seed = 5
    af.n_prop_steps = 100
    af.n_blocks = 1
    af.walker_kind = "unrestricted"
    af.build_job()

    job = af._job

    from trot.meas.ucisd import energy_kernel_gw_rh
    from trot.trial.ucisd import overlap_g
    from trot.core.ops import k_energy
    from trot.spin_proj import make_overlap_u_s2, make_energy_kernel_uw_rh_s2

    # Spin projection
    ## Data for the quadrature
    target_spin = 0
    betas, w_betas = trot.spin_proj.quadrature_s2(
        target_spin,
        (job.sys.nup, job.sys.ndn),
        4,
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

    e, err = af.kernel()

    assert jnp.isclose(e[-1].real, e_ref), (e, e_ref)
    assert jnp.isclose(err[-1].real, err_ref), (err, err_ref)
