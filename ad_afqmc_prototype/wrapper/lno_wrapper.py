import numpy as np
from functools import partial

from .. import driver #lnodriver
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.rhf import overlap_r,make_build_lno_meas_ctx,force_bias_kernel_rw_rh, energy_kernel_rw_rh, lnoenergy_kernel_rw_rh
from ..prop.afqmc import make_prop_ops
from ..prop.blocks import block
from ..prop.types import QmcParams
from ..trial.rhf import RhfTrial, make_rhf_trial_ops
from ..core.ops import MeasOps, k_energy, k_force_bias, o_orb_corr
from ..meas.rhf import make_build_lno_meas_ctx
from ..stat_utils import blocking_analysis_ratio, reject_outliers
from ..config import setup_jax
import numpy as np

import jax.numpy as jnp

class LnoRhf: #Right now only works for RHF trial and walkers
    def __init__(
        self,
        mf,
        prjlo=None,
        dt=0.005,
        seed=None,
        n_eql_blocks=50,
        n_blocks=500,
        mo_coeff=None,
        h0=None,
        h1=None,
        chol=None,
        n_chunks=1,
        n_walkers=200,
        target_error=0.001,
        use_gpu=False,
        single_precision=False,
        quiet=False,
    ):
        self.quiet = quiet
        self.use_gpu = use_gpu
        self.single_precision = single_precision
        setup_jax(use_gpu=self.use_gpu, single_precision=self.single_precision, quiet=self.quiet)

        self.mf = mf
        self.mol = mf.mol

        # store scalar params
        self.n_eql_blocks = n_eql_blocks
        self.n_blocks = n_blocks
        self.seed = seed if seed is not None else np.random.randint(0, int(1e6))
        self.dt = dt
        self.n_chunks = n_chunks
        self.n_walkers = n_walkers

        self.prjlo = None if prjlo is None else jnp.asarray(prjlo)
        self.h0 = None if h0 is None else jnp.asarray(h0)
        self.h1 = None if h1 is None else jnp.asarray(h1)
        self.chol = None if chol is None else jnp.asarray(chol)
        self.mo_coeff = None if mo_coeff is None else jnp.asarray(mo_coeff)
        self.target_error = target_error  
          

        self._built = False

    # def build_job_lno(self,prjlo,mo_coeff,h0,h1,chol):
    #     self.params = QmcParams(
    #         n_eql_blocks=self.n_eql_blocks,
    #         n_blocks=self.n_blocks,
    #         seed=self.seed,
    #         n_walkers=self.n_walkers,
    #         dt=self.dt,
    #         n_chunks=self.n_chunks,
    #         # n_exp_terms=self.n_exp_terms,
    #         # pop_control_damping=self.pop_control_damping,
    #         # weight_floor=self.weight_floor,
    #         # weight_cap=self.weight_cap,
    #         # n_prop_steps=self.n_prop_steps,
    #         # shift_ema=self.shift_ema,
    #     )
    #     from ..staging import StagedInputs, HamInput,TrialInput
    #     from ..setup import setup as setup_job
    #     ham = HamInput(h0 = self.h0,
    #                    h1 = np.asarray(self.h1),
    #                    chol=np.asarray(chol),
    #                    nelec=(mo_coeff.shape[1],mo_coeff.shape[1]),
    #                    norb=mo_coeff.shape[0],
    #                    chol_cut = 1e-5,
    #                    norb_frozen=0,
    #                    source_kind = "rhf",
    #                    basis="restricted") # type: ignore
    #     self.sys = System(norb=mo_coeff.shape[0], nelec=[mo_coeff.shape[1], mo_coeff.shape[1]], walker_kind="restricted")
    #     trial = TrialInput(kind="rhf",data={"mo":make_rhf_trial_ops(sys=self.sys)},norb_frozen=0,source_kind="rhf")
    #     meta = {}
    #     self.staged = StagedInputs(ham=ham,trial=trial,meta=meta) # type: ignore
    #     job = setup_job(self.staged,
    #                     walker_kind = "rhf",
    #                     mixed_precision=False,
    #                     params = self.params,
    #                     trial_data=trial,
    #                     trial_ops = make_rhf_trial_ops(sys=self.sys),
    #                     meas_ops = MeasOps(
    #                             overlap=overlap_r,
    #                             build_meas_ctx= make_build_lno_meas_ctx(self.prjlo),
    #                             kernels={
    #                                 k_force_bias: force_bias_kernel_rw_rh,
    #                                 k_energy: energy_kernel_rw_rh,
    #                             },
    #                             observables={
    #                                 o_orb_corr: lnoenergy_kernel_rw_rh,
    #                             }
    #                         ),
    #                     prop_ops = make_prop_ops(ham_basis="restricted", walker_kind=self.sys.walker_kind,mixed_precision = False),
    #                     block_fn = block
    #                     )
    #     return job


    def setup(self, prjlo, mo_coeff, h0, h1, chol):
        """Build everything that depends on prjlo/mo_coeff/integrals."""
        if prjlo is None:
            raise ValueError("setup() requires prjlo (array-like).")
        if mo_coeff is None:
            raise ValueError("setup() requires mo_coeff (array-like).")
        if h0 is None or h1 is None or chol is None:
            raise ValueError("setup() requires h0, h1, chol (array-like).")

        self.prjlo = jnp.asarray(prjlo)
        self.mo_coeff = jnp.asarray(mo_coeff)
        self.h0 = jnp.asarray(h0)
        self.h1 = jnp.asarray(h1)
        self.chol = jnp.asarray(chol)
        norb = mo_coeff.shape[0]
        nelec = mo_coeff.shape[1]

        self.sys = System(norb=norb, nelec=[nelec, nelec], walker_kind="restricted")
        self.ham_data = HamChol(self.h0, self.h1, self.chol)
        self.trial_data = RhfTrial(self.mo_coeff[:, : nelec])
        self.trial_ops = make_rhf_trial_ops(sys=self.sys)
        self.meas_ops = MeasOps(
            overlap=overlap_r,
            build_meas_ctx= make_build_lno_meas_ctx(self.prjlo),
            kernels={
                k_force_bias: force_bias_kernel_rw_rh,
                k_energy: energy_kernel_rw_rh,
            },
            observables={
                o_orb_corr: lnoenergy_kernel_rw_rh,
            }
        )


        self.prop_ops = make_prop_ops(ham_basis="restricted", walker_kind=self.sys.walker_kind,mixed_precision = False)

        self.params = QmcParams(
            n_eql_blocks=self.n_eql_blocks,
            n_blocks=self.n_blocks,
            seed=self.seed,
            n_walkers=self.n_walkers,
            dt=self.dt,
            n_chunks=self.n_chunks,
        )
        self.initial_walkers = None 
        self.block_fn = block
        return self


    def kernel(self):
        self.setup(self.prjlo, self.mo_coeff, self.h0, self.h1, self.chol)
        qmc_result = driver.run_qmc(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            target_error=self.target_error,
            observable_names=["orb_corr"]
        )
        orb_corr = qmc_result.observable_means["orb_corr"].real
        orb_corr_stderr = qmc_result.observable_stderrs["orb_corr"]
        print(f"Orbital correlation energy: {orb_corr:.6f} +/- {orb_corr_stderr:.6f}")

        return orb_corr, orb_corr_stderr

