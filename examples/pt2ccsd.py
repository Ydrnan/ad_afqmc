"""
Example: a manual setup of AFQMC/pt2CCSD energy for 8 non-interacting H2 dimers
================================================================================

This script demonstrates how to run an AFQMC calculation using pt2CCSD
trial wavefunction manually, without a high-level driver object.
AFQMC/pt2CCSD uses a perturbative CCSD wavefunction as the trial while the
guide (propagation) wavefunction remains at the RHF wavefunction.

workflow on setup AFQMC/pt2CCSD (Manually)
---------------------------------------------------------------------------------
1. Run a PySCF RHF + CCSD calculation to obtain MOs and amplitudes.
2. Stage the guide (RHF) Hamiltonian and the pt2CCSD trial separately.
3. Build operator factories (trial, measurement, propagation).
4. Run the mixed AFQMC via run_mixed_qmc().
5. Post-process the raw block data with clean_pt2ccsd() and blocking analysis.

Note: ad_afqmc_prototype does not yet have a single high-level interface for
pt2CCSD (unlike Afqmc for CISD).  All setup steps are therefore explicit here.
"""

from ad_afqmc_prototype import config

config.configure_once()

import jax.numpy as jnp
from pyscf import cc, gto, scf

from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.driver import run_mixed_qmc
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.pt2ccsd import (
    Pt2ccsdMeasCfg,
    build_meas_ctx,
    make_pt2ccsd_meas_ops,
)
from ad_afqmc_prototype.meas.rhf import make_rhf_meas_ops
from ad_afqmc_prototype.prop.afqmc import make_prop_ops
from ad_afqmc_prototype.prop.blocks import block_mixed
from ad_afqmc_prototype.prop.types import QmcParams
from ad_afqmc_prototype.staging import StagedMfOrCc, _stage_pt2ccsd_input, stage
from ad_afqmc_prototype.trial.pt2ccsd import Pt2ccsdTrial
from ad_afqmc_prototype.trial.rhf import RhfTrial, make_rhf_trial_ops

# =============================================================================
# Section 1: Molecular system
# =============================================================================
# Build a chain of nc H2 dimers, each with bond length a (Bohr) and
# inter-dimer separation d (Bohr).  The large separation makes this a
# weakly interacting cluster model, useful for benchmarking.

a = 2  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na = 2  # atoms per monomer (H2)
nc = 8  # number of monomers
elmt = "H"
unit = "b"  # length unit: Bohr
basis = "sto6g"

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=4)

# =============================================================================
# Section 2: Mean-field and coupled-cluster reference
# =============================================================================
# RHF provides the guide (propagation) wavefunction and the one- and
# two-electron integrals.  CCSD provides the t2 amplitudes used in the
# pt2CCSD trial wavefunction.

mf = scf.RHF(mol)
mf.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# Section 3: Stage the guide Hamiltonian (RHF)
# =============================================================================
# stage() converts the PySCF mean-field object into ad_afqmc's internal
# representation: a StagedInputs containing ham (integrals in the MO basis)
# and a trivial RHF trial.
#
# The Hamiltonian is built from the RHF MOs so that the guide propagator
# remains at the mean-field level.  The pt2CCSD trial is staged separately
# in the next section.

staged_guide = stage(mf)
ham = staged_guide.ham

# Construct the System descriptor: norb, nelec, and walker_kind ("restricted"
# means a single spin-up Slater determinant, sharing MOs with spin-down).
sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind="restricted")

# HamChol is the low-level JAX array container for the Hamiltonian.
# h0: scalar constant (nuclear repulsion + frozen-core energy)
# h1: (norb, norb) one-body matrix in the MO basis
# chol: (n_chol, norb, norb) Cholesky factors of the two-electron integrals
ham_data = HamChol(
    jnp.asarray(ham.h0),
    jnp.asarray(ham.h1),
    jnp.asarray(ham.chol),
    basis=ham.basis,
)

# =============================================================================
# Section 4: Build the guide and trial wavefunctions
# =============================================================================
# Guide wavefunction: RHF Slater determinant.
# mo_coeff holds the occupied MO columns (norb x nocc).

guide_data = RhfTrial(
    mo_coeff=jnp.array(staged_guide.trial.data["mo"][:, : sys.nup]),
)

# Trial wavefunction: pt2CCSD.
# _stage_pt2ccsd_input extracts the CC MO coefficients (mo_t) and the
# t2 amplitudes from the CCSD object and stores them in a StagedInputs.
# Pt2ccsdTrial is the low-level JAX array container for the trial.

trial_obj = StagedMfOrCc(mycc, norb_frozen=None)
staged_trial = _stage_pt2ccsd_input(trial_obj)

trial_data = Pt2ccsdTrial(
    mo_t=jnp.asarray(staged_trial.data["mo_t"]),
    t2=jnp.asarray(staged_trial.data["t2"]),
)

# =============================================================================
# Section 5: Build the measurement context for the pt2CCSD trial
# =============================================================================
# build_meas_ctx precomputes intermediate contractions (e.g. rotated Cholesky
# vectors in the trial MO basis) that are reused at every walker step.
# memory_mode="low" (currently doesn't matter in pt2CCSD trial)
# recomputing them on the fly — preferred for large systems.

trial_cfg = Pt2ccsdMeasCfg(memory_mode="low")
trial_meas_ctx = build_meas_ctx(ham_data, trial_data, trial_cfg)

# =============================================================================
# Section 6: Operator factories
# =============================================================================
# These factories return pure functions (no state) that are JIT-compiled by
# JAX the first time they are called.
#
#   guide_ops      : overlap and orbitals for the RHF guide
#   guide_meas_ops : mixed-estimator observables evaluated against the guide
#   guide_prop_ops : one-body + HS propagator acting on walker Slater dets
#   trial_meas_ops : pt2CCSD energy and weight estimators
#
# mixed_precision=False uses float64 throughout; set True for float32
# intermediates (faster on GPU, slightly less accurate).

guide_ops = make_rhf_trial_ops(sys)
guide_meas_ops = make_rhf_meas_ops(sys)
guide_prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
trial_meas_ops = make_pt2ccsd_meas_ops(sys, mixed_precision=False)

# =============================================================================
# Section 7: QMC parameters
# =============================================================================
# dt          : imaginary-time step (smaller = more accurate, more steps needed)
# n_walkers   : walker population (larger = less stochastic noise per block)
# n_prop_steps: propagation steps between measurements
# n_eql_blocks: equilibration blocks (discarded; let walkers reach steady state)
# n_blocks    : production blocks used for averaging

params = QmcParams(
    dt=0.005,
    n_walkers=200,
    n_prop_steps=50,
    n_blocks=200,
    n_eql_blocks=40,
    seed=17,
)

print(f"\ndt={params.dt}  n_walkers={params.n_walkers}  n_prop_steps={params.n_prop_steps}")
print(
    f"Equilibration imaginary time : {params.n_eql_blocks * params.n_prop_steps * params.dt:.2f} a.u."
)
print(
    f"Sampling imaginary time      : {params.n_blocks    * params.n_prop_steps * params.dt:.2f} a.u.\n"
)

# =============================================================================
# Section 8: Run AFQMC
# =============================================================================
# run_mixed_qmc propagates the guide walkers under the RHF Hamiltonian while
# evaluating pt2CCSD observables at each block.  It returns a MixedQmcResult
# containing per-block energies and weights for both the guide and the trial.

mixed_samples = run_mixed_qmc(
    sys=sys,
    params=params,
    ham_data=ham_data,
    guide_data=guide_data,
    guide_ops=guide_ops,
    guide_prop_ops=guide_prop_ops,
    guide_meas_ops=guide_meas_ops,
    trial_data=trial_data,
    trial_meas_ops=trial_meas_ops,
    mix_block_fn=block_mixed,
)

# =============================================================================
# Section 9: Extract and print final energies
# =============================================================================
# run_mixed_qmc already performs blocking analysis and outlier rejection
# internally and prints a summary.  The final values are also available on
# the returned object for downstream use.

print(
    f"\nGuide  (AFQMC/RHF)    : {mixed_samples.guide_mean_energy.real:.6f} +/- {mixed_samples.guide_stderr_energy.real:.6f} Ha"
)
print(
    f"Trial  (AFQMC/pt2CCSD): {mixed_samples.trial_mean_energy.real:.6f} +/- {mixed_samples.trial_stderr_energy.real:.6f} Ha"
)
print(f"Reference (CCSD)       : {mycc.e_tot:.6f} Ha")
