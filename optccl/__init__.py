"""OptCCL: Scalable Synthesis of Optimal Collective Communication Algorithms."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("optccl")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"

from .errors import OptcclError
from .config import Config, load_config, make_gurobi_env
from .topologies import BandwidthConstraint, Topology
from .topology_spec import TOPOLOGIES, load_or_build_topology
from .solver import (
    all_gather_lp,
    all_reduce_lp,
    all_to_all_lp,
    postprocess,
    reduce_scatter_lp,
)

__all__ = [
    "__version__",
    "OptcclError",
    "Config",
    "load_config",
    "make_gurobi_env",
    "BandwidthConstraint",
    "Topology",
    "TOPOLOGIES",
    "load_or_build_topology",
    "all_gather_lp",
    "reduce_scatter_lp",
    "all_reduce_lp",
    "all_to_all_lp",
    "postprocess",
]
