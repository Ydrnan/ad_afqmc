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
from pyscf import mcscf, ao2mo,lo
from ad_afqmc_prototype.staging import modified_cholesky

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

def run_afqmc_lno_mf(
    mf,
    norb_act=None,
    nelec_act=None,
    mo_coeff=None,
    norb_frozen=[],
    chol_cut=1e-5,
    seed=None,
    dt=0.005,
    n_walkers=5,
    nblocks=1000,
    target_error=1e-4,
    prjlo=None,
    n_eql=2,
):
    mol = mf.mol
    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    # calculate cholesky integrals
    # print("# Calculating Cholesky integrals")
    
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    DFbas = mf.with_df.auxmol.basis
    nbasis = mol.nao
    mc = mcscf.CASSCF(mf, norb_act, nelec_act)
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()
    nbasis = mo_coeff.shape[-1]
    # if norb_frozen == 0: norb_frozen = []
    act = [i for i in range(nbasis) if i not in norb_frozen]
    e = ao2mo.kernel(mf.mol, mo_coeff[:, act])#, compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)

    h1e = np.asarray(h1e)
    enuc = float(enuc)
    nbasis = h1e.shape[-1]
    chol = chol.reshape((-1, nbasis, nbasis))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))

    q = np.eye(mol.nao - len(norb_frozen))
    trial_coeffs[0] = q
    trial_coeffs[1] = q
    mo_coeff = trial_coeffs
    from ad_afqmc_prototype.wrapper.lno_wrapper import LnoRhf
    myafqmc = LnoRhf(mf,
        n_eql_blocks=n_eql,
        n_blocks=nblocks,
        seed=seed,
        dt=dt,
        n_walkers=n_walkers,
        prjlo=prjlo,
        target_error=target_error,
        h0 =enuc,
        h1 = h1e,
        chol = chol,
        mo_coeff = mo_coeff[0][:,:nelec[0]],

    )
    mean_ecorr,err_ecorr = myafqmc.kernel()
    return mean_ecorr, err_ecorr 

def prep_local_orbitals(mf, frozen=0, localization_method="pm"):
    if localization_method not in ["pm"]:
        raise ValueError(
            f"Localization method '{localization_method}' is not supported. Make LOs by yourself."
        )
    orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
    # lo_coeff = lo.PipekMezey(mf.mol, orbocc).kernel()
    mlo = lo.PipekMezey(mf.mol, orbocc)
    lo_coeff = mlo.kernel()
    while (
        True
    ):  # always performing jacobi sweep to avoid trapping in local minimum/saddle point
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=2)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    # # Fragment list: for PM, every orbital corresponds to a fragment
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    return lo_coeff, frag_lolist


