from __future__ import annotations

from ..core.ops import MeasOps
from ..core.system import System
from ..trial.ucisdtq import UcisdtqTrial
from .auto import make_auto_meas_ops


def make_ucisdtq_meas_ops(sys: System, trial_ops) -> MeasOps:
    """
    Returns auto (finite-difference) measurement ops for UCISDTQ.

    No manual force-bias or energy kernels exist for the quadruples sector,
    so this delegates to make_auto_meas_ops which differentiates the overlap.
    """
    return make_auto_meas_ops(sys, trial_ops)
