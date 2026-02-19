from __future__ import annotations

import os
import platform
import socket
import sys
import warnings
from dataclasses import dataclass
from typing import Optional


def is_jupyter_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


@dataclass
class AfqmcConfig:
    """
    Global configuration.

    use_gpu:
      - None  : auto (prefer GPU if available, else CPU)
      - True  : force GPU (error if unavailable)
      - False : force CPU
    """

    use_gpu: Optional[bool] = None
    single_precision: bool = False
    quiet: bool = True  # suppress prints


afqmc_config = AfqmcConfig()

_configured_once = False


def configure_once(
    *,
    use_gpu: Optional[bool] = None,
    single_precision: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> None:
    """
    Configure JAX once, subsequent calls do nothing.
    Use GPU if available by default.
    """
    global _configured_once
    if _configured_once:
        return

    if use_gpu is not None:
        afqmc_config.use_gpu = use_gpu
    if single_precision is not None:
        afqmc_config.single_precision = single_precision
    if quiet is not None:
        afqmc_config.quiet = quiet

    setup_jax(
        use_gpu=afqmc_config.use_gpu,
        single_precision=afqmc_config.single_precision,
        quiet=afqmc_config.quiet,
    )
    _configured_once = True


def setup_jax(*, use_gpu: Optional[bool], single_precision: bool, quiet: bool) -> None:
    """
    Configure JAX runtime.
    """
    # if JAX is already imported, some settings may be too late.
    if "jax" in sys.modules:
        warnings.warn(
            "JAX was imported before AFQMC configuration; some JAX/XLA settings may not take effect. "
            "For full control, call config.configure_once(...) before importing jax.",
            stacklevel=2,
        )

    # prefer x64 unless single_precision is requested
    if not single_precision:
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    if use_gpu is True:
        os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    elif use_gpu is False:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_force_host_platform_device_count=1 "
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        )

    from jax import config as jax_config

    # breaking change in random number generation in jax v0.5
    jax_config.update("jax_threefry_partitionable", False)

    if use_gpu is True:
        jax_config.update("jax_platform_name", "gpu")
    elif use_gpu is False:
        jax_config.update("jax_platform_name", "cpu")

    import jax

    platforms = {d.platform for d in jax.devices()}
    got_gpu = "gpu" in platforms

    if use_gpu is None:
        afqmc_config.use_gpu = got_gpu
    else:
        afqmc_config.use_gpu = bool(use_gpu)

    if use_gpu is True and not got_gpu:
        raise RuntimeError(
            "use_gpu=True requested, but JAX did not initialize a GPU backend."
        )

    if (not quiet) and got_gpu:
        _print_host_info()


def _print_host_info() -> None:
    hostname = socket.gethostname()
    uname_info = platform.uname()
    print(f"# Hostname: {hostname}")
    print("# Using GPU (Policy A).")
    print(f"# System: {uname_info.system}")
    print(f"# Node Name: {uname_info.node}")
    print(f"# Release: {uname_info.release}")
    print(f"# Version: {uname_info.version}")
    print(f"# Machine: {uname_info.machine}")
    print(f"# Processor: {uname_info.processor}")
