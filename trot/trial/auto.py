from __future__ import annotations

from ..core.ops import OverlapFn, Rdm1Fn, TrialOps
from ..core.system import System


def _require_overlap(
    overlap: OverlapFn | None,
    *,
    walker_kind: str,
    param_name: str,
) -> OverlapFn:
    if overlap is None:
        raise ValueError(
            f"auto trial for walker_kind='{walker_kind}' requires `{param_name}` to be provided"
        )
    return overlap


def make_auto_trial_ops(
    sys: System,
    *,
    overlap_r: OverlapFn | None = None,
    overlap_u: OverlapFn | None = None,
    overlap_g: OverlapFn | None = None,
    get_rdm1: Rdm1Fn,
) -> TrialOps:
    """
    Factory for overlap only trial prototypes.
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return TrialOps(
            overlap=_require_overlap(overlap_r, walker_kind=wk, param_name="overlap_r"),
            get_rdm1=get_rdm1,
        )

    if wk == "unrestricted":
        return TrialOps(
            overlap=_require_overlap(overlap_u, walker_kind=wk, param_name="overlap_u"),
            get_rdm1=get_rdm1,
        )

    if wk == "generalized":
        return TrialOps(
            overlap=_require_overlap(overlap_g, walker_kind=wk, param_name="overlap_g"),
            get_rdm1=get_rdm1,
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
