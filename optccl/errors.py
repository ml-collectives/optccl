class OptcclError(Exception):
    """A user-facing error: the message is shown verbatim, without a traceback."""


class RmpInfeasibleError(OptcclError):
    """Phase 1 of a DW solve converged with residual slack: the requested ``R``
    is below the collective's achievable rate on this topology. Subclasses
    ``OptcclError`` so the CLI still prints it cleanly, but is distinct so the
    rate-retry loop in ``mirrored_dw`` can catch exactly this case (and only
    this case) to bump ``R`` and re-solve."""
