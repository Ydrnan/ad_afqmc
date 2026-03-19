from __future__ import annotations

from .config import configure_once

configure_once()

import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray, ArrayLike

print = partial(print, flush=True)

from .core.system import WalkerKind
from .prop.types import QmcParamsBase, QmcParams, QmcParamsFp, QmcParamsLno
from .driver import QmcResult
from .setup import Job
from .setup import setup as setup_job
from .setup_fp import JobFp
from .setup_fp import setup_fp as setup_job_fp
from .setup_lno import setup_lno as setup_job_lno
from . import setup_lno
from .staging import StagedInputs, _is_cc_like
from .staging import dump as dump_staged
from .staging import load as load_staged
from .staging import stage as stage_inputs


def _default_seed() -> int:
    return int(np.random.randint(0, int(1e9)))


def banner_afqmc() -> str:
    return r"""
 █████╗ ██████╗        █████╗ ███████╗ ██████╗ ███╗   ███╗ ██████╗
██╔══██╗██╔══██╗      ██╔══██╗██╔════╝██╔═══██╗████╗ ████║██╔════╝
███████║██║  ██║█████╗███████║█████╗  ██║   ██║██╔████╔██║██║
██╔══██║██║  ██║╚════╝██╔══██║██╔══╝  ██║▄▄ ██║██║╚██╔╝██║██║
██║  ██║██████╔╝      ██║  ██║██║     ╚██████╔╝██║ ╚═╝ ██║╚██████╗
╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝╚═╝      ╚══▀▀═╝ ╚═╝     ╚═╝ ╚═════╝
     differentiable auxiliary-field quantum Monte Carlo
"""


class Afqmc:
    """
    AFQMC driver object.

    Parameters
    ----------
    mf_or_cc : Any
        Mean-field or coupled-cluster object from which to build Hamiltonian and trial wavefunction.
    norb_frozen : int, optional
        Number of orbitals to freeze (from the bottom), by default 0 or cc.frozen
        if mf_or_cc is a Pyscf SCF or CC instance, respectively. For CC instances,
        norb_frozen cannot be set to a value differing fron cc.frozen.
    chol_cut : float, optional
        Cholesky decomposition cutoff, by default 1e-5
    cache : Union[str, Path], optional
        Path to cache file for staged inputs, by default None
    n_eql_blocks : int, optional
        Number of equilibration blocks if params is not provided, by default 20
    n_blocks : int, optional
        Number of production blocks if params is not provided, by default 200
    seed : int | None, optional
        Random seed if params is not provided, by default None
    dt : float | None, optional
        Time step if params is not provided, by default None
    n_walkers : int | None, optional
        Number of walkers if params is not provided, by default None
    n_chunk : int | None, optional
        Number of chunks if params is not provided, by default 1
    """

    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | ArrayLike | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        self._obj = mf_or_cc
        self._cc: Any = None
        if _is_cc_like(mf_or_cc):
            self._cc = mf_or_cc
            self._scf = mf_or_cc._scf
            self.source_kind = "cc"
        else:
            self._scf = mf_or_cc
            self.source_kind = "mf"

        self.norb_frozen = norb_frozen
        self.chol_cut = float(chol_cut)
        self.cache = Path(cache).expanduser().resolve() if cache is not None else None
        self.overwrite_cache = False
        self.verbose = False

        self.walker_kind: WalkerKind | None = None  # resolved in kernel
        self.mixed_precision = True

        self.params: QmcParamsBase | None = None  # resolved in kernel
        _defaults = QmcParams()
        self.dt = _defaults.dt if dt is None else dt
        self.n_walkers = _defaults.n_walkers if n_walkers is None else n_walkers
        self.n_blocks = _defaults.n_blocks if n_blocks is None else n_blocks
        self.n_eql_blocks = _defaults.n_eql_blocks if n_eql_blocks is None else n_eql_blocks
        self.seed = _defaults.seed if seed is None else seed
        self.n_chunks = _defaults.n_chunks if n_chunks is None else n_chunks

        self._staged: StagedInputs | None = None
        self._job: Job | None = None
        self._cache_key: tuple | None = None

        self.e_tot: Any = None
        self.e_err: Any = None
        self.block_energies: Any = None
        self.block_weights: Any = None

    @property
    def staged(self) -> StagedInputs | None:
        return self._staged

    @property
    def job(self) -> Job | None:
        return self._job

    def _dump_params(self, params: QmcParams) -> None:
        assert isinstance(
            params, QmcParams
        ), f"Expected a QmcParams instance, but got {type(params)}"
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        print(" QmcParams:")
        for field in fields:
            print(f"  {field.name:<{width}} = {getattr(params, field.name)}")
        print("")

    def dump_flags(self, job) -> None:
        assert isinstance(job, Job), f"Expected a Job instance, but got {type(job)}"
        self._dump_flags_helper(job)

    def _dump_flags_helper(self, job) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.staged.ham.chol.shape[0]
        params = job.params
        trial = job.staged.trial
        print("******** AFQMC ********")
        print(f" norb            = {sys.norb}")
        print(f" nelec_up        = {sys.nelec[0]}")
        print(f" nelec_dn        = {sys.nelec[1]}")
        print(f" nchol           = {nchol}")
        print(f" source_kind     = {src}")
        print(f" trial_kind      = {trial.kind}")
        print(f" chol_cut        = {chol_cut:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" mixed_precision = {self.mixed_precision}\n")
        self._dump_params(params)

    def _key(self) -> tuple:
        """Key for determining whether staged/job caches are still valid."""
        cache_mtime = None
        if self.cache is not None and self.cache.exists():
            cache_mtime = self.cache.stat().st_mtime
        return (
            self.source_kind,
            self.norb_frozen,
            float(self.chol_cut),
            str(self.cache) if self.cache is not None else None,
            bool(self.overwrite_cache),
            cache_mtime,
        )

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        staged = stage_inputs(
            self._obj,
            norb_frozen=self.norb_frozen if self.norb_frozen is not None else None,
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
        )
        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def save_staged(self, path: Union[str, Path]) -> None:
        """Write current staged inputs to a single file cache."""
        staged = self.stage()
        dump_staged(staged, path)

    # def load_staged(self, path: Union[str, Path]): -> StagedInputs:
    #    """Load staged inputs from a cache file and attach them to this object."""
    #    staged = load_staged(path)
    #    self._staged = staged
    #    self._cache_key = None
    #    self._job = None
    #    return staged

    def _make_params(self) -> QmcParams:
        """
        Create QmcParams if user didn't provide one.
        """
        if self.params is not None and isinstance(self.params, QmcParams):
            params = self.params
        elif self.params is not None and not isinstance(self.params, QmcParams):
            raise TypeError(
                f"Expected type QmcParams for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(QmcParams):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = QmcParams(**kwargs)

        return params

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if self._job is not None and not force:
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = setup_job(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=qmc_params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[float, float]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        job = self.build_job()
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)

        e_tot = float(qmc_result.mean_energy)
        e_err = float(qmc_result.stderr_energy)

        self.qmc_result = qmc_result

        return e_tot, e_err

    run = kernel

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> Afqmc:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen orbitals, norb_frozen,
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        staged = load_staged(path)
        meta = staged.meta

        mf_or_cc = None

        # Cannot be changed as the input has been staged
        norb_frozen = meta["norb_frozen"]
        chol_cut = meta["chol_cut"]

        af = Afqmc(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()

        return af


class AfqmcFp(Afqmc):
    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | ArrayLike | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
        ene0: float | None = None,
        n_traj: int | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=None,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        self.n_traj = n_traj
        self.ene0 = ene0

    def _dump_params(self, params: QmcParamsFp) -> None:
        assert isinstance(
            params, QmcParamsFp
        ), f"Expected a QmcParamsFp instance, but got {type(params)}"
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        print(" QmcParamsFp:")
        for field in fields:
            print(f"  {field.name:<{width}} = {getattr(params, field.name)}")
        print("")

    def dump_flags(self, job) -> None:
        assert isinstance(job, JobFp), f"Expected a JobFp instance, but got {type(job)}"
        self._dump_flags_helper(job)

    def _make_params(self) -> QmcParamsFp:
        """
        Create QmcParamsFp if user didn't provide one.
        """
        if self.params is not None and isinstance(self.params, QmcParamsFp):
            params = self.params
        elif self.params is not None and not isinstance(self.params, QmcParamsFp):
            raise TypeError(
                f"Expected type QmcParamsFp for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(QmcParamsFp):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = QmcParamsFp(**kwargs)

        if params.ene0 is None:
            raise ValueError(
                "The value of the parameter 'ene0' must be set, typically with SCF or CC energy."
            )

        return params

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
    ) -> JobFp:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if self._job is not None and not force:
            assert isinstance(self._job, JobFp)
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = setup_job_fp(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=qmc_params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def kernel(self, **driver_kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        job = self.build_job()
        self.dump_flags(job)

        qmc_result = job.kernel(**driver_kwargs)

        e_tot = np.asarray(qmc_result.mean_energy)
        e_err = np.asarray(qmc_result.stderr_energy)

        self.qmc_result = qmc_result

        return e_tot, e_err

    run_fp = kernel

    @classmethod
    def from_staged(
        cls,
        path: Union[str, Path],
        *,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int = 1,
    ) -> AfqmcFp:
        """
        Returns a new AFQMC object from a previously staged calculations
        (using save_staged method). The number of frozen orbitals, norb_frozen,
        and the choliesky decomposition threshold, chol_cut, cannot be changed.
        Parameters
        ----------
        path: str, pathlib.Path
        The other parameters are identical to the ones in the AFQMC class.
        """
        staged = load_staged(path)
        meta = staged.meta

        mf_or_cc = None

        # Cannot be changed as the input has been staged
        norb_frozen = meta["norb_frozen"]
        chol_cut = meta["chol_cut"]

        af = AfqmcFp(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        af._staged = staged
        af.source_kind = meta["source_kind"]
        af._cache_key = af._key()

        return af


class AfqmcLnoFrag(Afqmc):
    def __init__(
        self,
        mf_or_cc: Any,
        *,
        norb_frozen: int | ArrayLike | None = None,
        chol_cut: float = 1e-5,
        cache: Union[str, Path] | None = None,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
        prjlo: NDArray | None = None,
    ):
        super().__init__(
            mf_or_cc,
            norb_frozen=norb_frozen,
            chol_cut=chol_cut,
            cache=cache,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )

        self.mixed_precision = False
        self.prjlo = prjlo

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        Compute or load HamInput/TrialInput.
        If cache is set and exists, loads unless overwrite_cache=True.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        assert isinstance(self.norb_frozen, (tuple, list, np.ndarray))
        ham = setup_lno.build_ham(
            self._obj,
            norb_frozen=self.norb_frozen,
            chol_cut=self.chol_cut,
        )

        staged = stage_inputs(
            self._obj,
            norb_frozen=self.norb_frozen if self.norb_frozen is not None else None,
            chol_cut=self.chol_cut,
            cache=self.cache,
            overwrite=self.overwrite_cache if self.cache is not None else False,
            verbose=self.verbose,
            ham=ham,
            trial=None,
        )

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    def build_job(
        self,
        *,
        force: bool = False,
        trial_data: Any = None,
        trial_ops: Any = None,
        meas_ops: Any = None,
        prop_ops: Any = None,
        block_fn: Callable[..., Any] | None = None,
        prop_kwargs: dict[str, Any] | None = None,
    ) -> Job:
        """
        Assemble a runnable Job from current settings and staged inputs.
        """
        if self._job is not None and not force:
            return self._job

        staged = self.stage()
        qmc_params = self._make_params()
        self.params = qmc_params

        job = setup_job_lno(
            staged,
            walker_kind=self.walker_kind,
            mixed_precision=self.mixed_precision,
            params=qmc_params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            prop_kwargs=prop_kwargs,
        )
        self._job = job
        return job

    def _make_params(self) -> QmcParamsLno:
        """
        Create QmcParams if user didn't provide one.
        """
        if self.params is not None and isinstance(self.params, QmcParamsLno):
            params = self.params
        elif self.params is not None and not isinstance(self.params, QmcParamsLno):
            raise TypeError(
                f"Expected type QmcParamsLno for self.params, but received '{type(self.params)}'"
            )
        else:
            kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(QmcParamsLno):
                if hasattr(self, field.name):
                    val = getattr(self, field.name)
                    if val is not None:
                        kwargs[field.name] = val

            params = QmcParamsLno(**kwargs)

        return params

    def _dump_params(self, params: QmcParamsLno) -> None:
        assert isinstance(
            params, QmcParamsLno
        ), f"Expected a QmcParamsLno instance, but got {type(params)}"
        fields = dataclasses.fields(params)
        width = len(max(fields, key=lambda f: len(f.name)).name)
        print(" QmcParamsLno:")
        for field in fields:
            print(f"  {field.name:<{width}} = {getattr(params, field.name)}")
        print("")

    def _dump_flags_helper(self, job) -> None:
        meta = job.staged.meta
        src = meta["source_kind"]
        chol_cut = meta["chol_cut"]
        sys = job.sys
        nchol = job.staged.ham.chol.shape[0]
        params = job.params
        trial = job.staged.trial
        print("******** AFQMC ********")
        print(f" nchol           = {nchol}")
        print(f" source_kind     = {src}")
        print(f" trial_kind      = {trial.kind}")
        print(f" chol_cut        = {chol_cut:g}")
        print(f" cache           = {str(self.cache) if self.cache else None}")
        print(f" walker_kind     = {sys.walker_kind}")
        print(f" mixed_precision = {self.mixed_precision}\n")
        self._dump_params(params)

    def kernel(self, **driver_kwargs: Any) -> tuple[NDArray, NDArray]:
        """
        Runs AFQMC, returns (e_tot, e_err), and stores samples.
        """
        print(banner_afqmc())
        job = self.build_job()
        self.dump_flags(job)

        obs = driver_kwargs.get("observable_names", ())
        if "orb_corr" not in obs:
            driver_kwargs["observable_names"] = obs + ("orb_corr",)

        qmc_result = job.kernel(prop=True, **driver_kwargs)

        if not isinstance(qmc_result, QmcResult):
            raise TypeError(
                f"Unexpected return from Job.kernel(), expected QmcResult but received {type(qmc_result)}."
            )

        orb_corr = np.array(qmc_result.observable_means["orb_corr"].real)
        orb_corr_stderr = np.array(qmc_result.observable_stderrs["orb_corr"])

        self.qmc_result = qmc_result

        return orb_corr, orb_corr_stderr


# Backward-compatible aliases
AFQMC = Afqmc
AFQMCFp = AfqmcFp
