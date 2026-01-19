import os
import pickle
import shlex
import subprocess
from functools import partial
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from pyscf import __config__, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import config, driver, grad_utils, optimize_trial, utils, wavefunctions

from ad_afqmc.prep import PrepAfqmc

print = partial(print, flush=True)


def run_afqmc(
    options: Optional[dict] = None,
    mpi_prefix: Optional[str] = None,
    nproc: Optional[int] = None,
    tmpdir: str = ".",
):
    """
    Run AFQMC calculation from pre-generated input files.

    Parameters:
        options : dict, optional
            Options for AFQMC.
        mpi_prefix : str, optional
            MPI prefix, used to launch MPI processes.
        nproc : int, optional
            Number of processes, if using MPI.
        tmpdir : str, optional
            Temporary directory where the input files are stored.
    """
    if options is not None:
        with open(tmpdir + "/options.bin", "wb") as f:
            pickle.dump(options, f)
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    script = f"{dir_path}/launch_script.py"
    use_gpu = config.afqmc_config["use_gpu"]
    use_mpi = config.afqmc_config["use_mpi"]

    if not use_gpu and config.afqmc_config["use_mpi"] is not False:
        try:
            from mpi4py import MPI

            if not MPI.Is_finalized():
                MPI.Finalize()
            use_mpi = True
            print(f"# mpi4py found, using MPI.")
            if nproc is None:
                print(f"# Number of MPI ranks not specified, using 1 by default.")
                nproc = 1
        except ImportError:
            use_mpi = False
            if mpi_prefix is not None or nproc is not None:
                raise ValueError(
                    f"# MPI prefix or number of processes specified, but mpi4py not found. Please install mpi4py or remove the MPI options."
                )
            else:
                print(f"# Unable to import mpi4py, not using MPI.")

    gpu_flag = "--use_gpu" if use_gpu else ""
    mpi_flag = "--use_mpi" if use_mpi else ""
    if mpi_prefix is None:
        if use_mpi:
            mpi_prefix = "mpirun "
            if nproc is not None:
                mpi_prefix += f"-np {nproc} "

        else:
            mpi_prefix = ""
    elif nproc is not None:
        mpi_prefix += f"-np {nproc}"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = shlex.split(f"{mpi_prefix} python {script} {tmpdir} {gpu_flag} {mpi_flag}")
    # Launch process with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
        bufsize=1,
    )
    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output, end="")
    return_code = process.poll()
    if return_code != 0:
        if return_code is None:
            return_code = -1
        raise subprocess.CalledProcessError(return_code, cmd)

    try:
        ene_err = np.loadtxt(tmpdir + "/ene_err.txt")
    except:
        print("AFQMC did not execute correctly.")
        ene_err = 0.0, 0.0
    return ene_err[0], ene_err[1]


def run_afqmc_ph(prep):
    """
    Run AFQMC calculation from pre-generated input files.

    Parameters:
        options : dict, optional
            Options for AFQMC.
        mpi_prefix : str, optional
            MPI prefix, used to launch MPI processes.
        nproc : int, optional
            Number of processes, if using MPI.
        tmpdir : str, optional
            Temporary directory where the input files are stored.
    """
    config.setup_jax()
    comm = config.setup_comm()

    prep.setup_afqmc()
    (
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        _,
        _,
        sampler,
        observable,
        options,
    ) = prep.get_setup_data()

    e_afqmc, err_afqmc = driver.afqmc(
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        sampler,
        observable,
        options,
        comm,
    )
    return e_afqmc, err_afqmc


def run_afqmc_fp(prep):
    config.setup_jax()
    comm = config.setup_comm()

    prep.setup_afqmc()
    (
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
    ) = prep.get_setup_data()

    if (
        options["symmetry_projector"] == "s2"
        and isinstance(trial, wavefunctions.uhf)
        and options["optimize_trial"]
    ):
        orbitals = optimize_trial.optimize_trial(ham_data, trial, wave_data)
        wave_data["mo_coeff"] = orbitals
        nao = orbitals[0].shape[0]
        wave_data["rdm1"] = jnp.vstack(
            [orbitals[0] @ orbitals[0].T.conj(), orbitals[1] @ orbitals[1].T.conj()]
        ).reshape(-1, nao, nao)

    e_afqmc, err_afqmc, sign = driver.fp_afqmc(
        ham_data,
        ham,
        prop,
        trial,
        wave_data,
        trial_ket,
        wave_data_ket,
        sampler,
        observable,
        options,
        comm,
    )

    return e_afqmc, err_afqmc, sign

class AFQMC:
    """
    AFQMC class.

    Attributes:
        mf_or_cc : Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
            The mean-field or CCSD object.
        basis_coeff : None
            Basis coefficients used for AFQMC propagation.
        norb_frozen : int
            Number of frozen orbitals.
        chol_cut : float
            Cholesky cut-off.
        integrals : None
            Dictionary of integrals in an orthonormal basis, {"h0": enuc, "h1": h1e, "h2": eri}.
        mpi_prefix : None
            MPI prefix, used to launch MPI processes.
        nproc : int
            Number of processes, if using MPI.
        dt : float
            AFQMC propagation time step.
        n_walkers : int
            Number of walkers.
        n_prop_steps : int
            Number of propagation steps.
        n_ene_blocks : int
            Number of energy blocks.
        n_sr_blocks : int
            Number of stochastic reconfiguration blocks.
        n_blocks : int
            Number of blocks.
        n_ene_blocks_eql : int
            Number of energy blocks for equilibration.
        n_sr_blocks_eql : int
            Number of stochastic reconfiguration blocks for equilibration.
        seed : int
            Random seed.
        n_eql : int
            Number of equilibration blocks.
        ad_mode : None
            AD mode.
        orbital_rotation : bool
            Orbital rotation option for AD.
        do_sr : bool
            Stochastic reconfiguration, relevant for AD.
        walker_type : str
            Walker type, "restricted" or "unrestricted".
        symmetry : bool
            Symmetry, relevant for AD.
        save_walkers : bool
            Save walkers.
        trial : str
            Trial.
        ene0 : float
            Initial energy used in free projection.
        n_chunks : int
            Number of chunks, relevant for GPU calculations.
        vhs_mixed_precision : bool
            Use mixed precision for VHS.
        trial_mixed_precision : bool
            Use mixed precision for trial (CISD).
        memory_mode: str
            Memory mode, "high" or "low" (CISD).
        tmpdir : str
            Temporary directory.
        prjlo: bool
            Used in LNO, need to fix.
    """

    def __init__(
        self,
        mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD],
        mf_or_cc_ket: Optional[
            Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]
        ] = None,
        mode: Optional[str] = "small",
    ):
        self.mode = mode
        self.mf_or_cc = mf_or_cc
        self.mf_or_cc_ket = mf_or_cc_ket if mf_or_cc_ket is not None else mf_or_cc
        self.basis_coeff = None
        frozen = getattr(mf_or_cc, "frozen", 0)
        if isinstance(frozen, int):
            self.norb_frozen = frozen
        else:
            print("Warning: Frozen is not an integer, assuming 0 frozen orbitals.")
            self.norb_frozen = 0
        self.chol_cut = 1e-5
        self.integrals = None  # custom integrals
        self.mpi_prefix = None
        self.nproc = 1

        self.set_trial_bra_label(mf_or_cc)
        self.set_trial_ket_label(mf_or_cc_ket)

        self.tmpdir = __config__.TMPDIR + f"/afqmc{np.random.randint(1, int(1e6))}/"

        # Set default options
        for attr, val in Options(mode).to_dict().items():
            if not hasattr(self, attr):
                setattr(self, attr, val)

    def set_trial_bra_label(self, mf_or_cc):
        self.trial = None
        if isinstance(mf_or_cc, scf.uhf.UHF) or isinstance(mf_or_cc, scf.rohf.ROHF):
            self.trial = "uhf"
        elif isinstance(mf_or_cc, scf.rhf.RHF):
            self.trial = "rhf"
        elif isinstance(mf_or_cc, UCCSD):
            self.trial = "ucisd"
        elif isinstance(mf_or_cc, CCSD):
            self.trial = "cisd"

    def set_trial_ket_label(self, mf_or_cc_ket):
        if isinstance(mf_or_cc_ket, scf.uhf.UHF) or isinstance(
            mf_or_cc_ket, scf.rohf.ROHF
        ):
            self.trial_ket = "uhf"
        elif isinstance(mf_or_cc_ket, scf.rhf.RHF):
            self.trial_ket = "rhf"
        elif isinstance(mf_or_cc_ket, UCCSD):
            self.trial_ket = "uccsd"
        elif isinstance(mf_or_cc_ket, CCSD):
            self.trial_ket = "ccsd"
        else:
            self.trial_ket = self.trial

    def make_options_dict(self):
        options = {}
        for attr in dir(self):
            if (
                attr
                not in [
                    "mode",
                    "mf_or_cc",
                    "mf_or_cc_ket",
                    "basis_coeff",
                    "norb_frozen",
                    "chol_cut",
                    "integrals",
                    "mpi_prefix",
                    "nproc",
                    "tmpdir",
                ]
                and not attr.startswith("__")
                and not callable(getattr(self, attr))
            ):
                options[attr] = getattr(self, attr)
        return options

    def kernel(self, dry_run=False):
        """
        Run AFQMC.

        Args:
            dry_run : bool
                If True, writes input files like integrals to disk, useful for large calculations where one would like to run the trial generation and AFQMC calculations on different machines, on CPU and GPU, for example.
        """
        os.makedirs(self.tmpdir, exist_ok=True)

        options = Options(self.mode)
        options.update_from_dict(self.make_options_dict())
        options = options.to_dict()

        if self.ad_mode == "nuc_grad":
            # grad_utils.prep_afqmc_nuc_grad(self.mf_or_cc, self.dR, tmpdir=self.tmpdir)
            raise NotImplementedError(
                "Nuclear gradients with AFQMC are not implemented yet."
            )

        config.setup_jax()
        prep = PrepAfqmc()
        prep.set_mol(self.mf_or_cc.mol)
        prep.set_pyscf_mf_cc(self.mf_or_cc, self.mf_or_cc_ket)
        prep.set_basis_coeff(self.basis_coeff)
        prep.set_frozen_core(self.norb_frozen)
        prep.set_chol_cut(self.chol_cut)
        prep.set_tmpdir(self.tmpdir)
        prep.options = options

        if dry_run or options["write_to_disk"]:
            prep.io.set_write()
        elif options["read_from_disk"]:
            prep.io.set_read()
        else:
            prep.io.set_noio()

        prep.prep()

        if dry_run:
            #with open("tmpdir.txt", "w") as f:
            #    f.write(self.tmpdir)
            #with open(self.tmpdir + "/options.bin", "wb") as f:
            #    pickle.dump(options, f)
            return self.tmpdir
        elif options["free_projection"]:
            return run_afqmc_fp(prep)
        else:
            return run_afqmc_ph(prep)


# I believe Options should be divided, it does not make sense to have all
# the keywords here as their number is becoming large.
# I don't think mode should be here.
# Default options from AFQMC class differ from the ones in utils...
class Options:
    # To catch typos...
    __slots__ = ("dt", "n_prop_steps", "n_ene_blocks",
    "n_walkers", "n_sr_blocks", "n_qr_blocks", "n_blocks", "n_ene_blocks_eql",
    "n_sr_blocks_eql", "n_eql", "seed", "ad_mode", "orbital_rotation", "trial", "trial_ket",
    "do_sr", "walker_type", "symmetry_projector", "ngrid", "optimize_trial",
    "s2_projector_ngrid", "target_spin", "symmetry", "save_walkers", "dR", "free_projection",
    "ene0", "n_chunks", "vhs_mixed_precision", "trial_mixed_precision",
    "memory_mode", "write_to_disk", "read_from_disk", "prjlo", "no_beta_0_fp"
    )

    def __init__(self, mode):
        self.dt = 0.005
        self.n_prop_steps = 50
        self.n_ene_blocks = 1
        self.n_qr_blocks = 1
        if mode == "small":
            self.n_walkers = 50
            self.n_sr_blocks = 1
            self.n_blocks = 200
            self.n_ene_blocks_eql = 1
            self.n_sr_blocks_eql = 5
            self.n_eql = 10
        elif mode == "production":
            self.n_walkers = 200
            self.n_sr_blocks = 20
            self.n_blocks = 500
            self.n_ene_blocks_eql = 5
            self.n_sr_blocks_eql = 10
            self.n_eql = 3
        self.seed = np.random.randint(1, int(1e6))
        self.ad_mode = None
        self.orbital_rotation = True
        self.do_sr = True
        self.walker_type = "restricted"
        self.trial = None
        self.trial_ket = None

        # this can be tr, s2 or sz for time-reversal, S^2, or S_z symmetry projection, respectively
        self.symmetry_projector = None
        self.ngrid = 4 # Number of grid point for the quadrature
        self.s2_projector_ngrid = 4 # Number of grid point for the quadrature
        self.optimize_trial = False
        self.target_spin = 0  # 2S and is only used when symmetry_projector is s2
        self.symmetry = False
        self.save_walkers = False
        self.dR = 1e-5  # displacement used in finite difference to calculate integral gradients for ad_mode = nuc_grad
        self.free_projection = False

        self.ene0 = 0.0
        self.n_chunks = 1
        self.vhs_mixed_precision = False
        self.trial_mixed_precision = False
        self.memory_mode = "low"
        self.write_to_disk = False # Write FCIDUMP and ci/cc coeff to disk
        self.read_from_disk = False # Read FCIDUMP and ci/cc coeff from disk
        self.prjlo = None  # used in LNO, need to fix
        self.no_beta_0_fp = False # Avoid computing the energy at 0 imaginary time in fp

    def to_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def update_from_dict(self, options: dict):
        t = type(options)
        if t != dict:
            raise TypeError(f"Expected a dict but received '{t}'.")
        for key, val in options.items():
            if key not in self.__slots__:
                raise KeyError(f"Keyword {key} does not exist.")
            setattr(self, key, val)
