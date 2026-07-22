from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import tomllib

from .errors import OptcclError

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    # Gurobi WLS credentials
    wls_access_id: str = "XXX"
    wls_secret: str = "XXX"
    license_id: int = 9999999

    # Subproblem solver settings
    ag_no_crossover: bool = True
    aa_no_crossover: bool = False
    ar_no_crossover: bool = True
    rs_no_crossover: bool = True

    # The *_phase1_cost_weight is the cost-awareness weight for Phase 1 of the
    # two-phase column generation (0 = pure feasibility); a small value steers
    # Phase 1 toward cost-relevant columns to improve convergence. Too large
    # can falsely cause infeasibility. Intended to be agnostic to input size.
    ag_phase1_cost_weight: float = 0.01
    ar_phase1_cost_weight: float = 0.01
    aa_phase1_cost_weight: float = 0.01
    rs_phase1_cost_weight: float = 0.01
    multi_phase1_cost_weight: float = 0.01

    # Stop the mirrored Dantzig-Wolfe solve as soon as Phase 1 reaches a feasible
    # RMP (all artificial slacks at zero), returning that feasible solution
    # without ever running Phase 2. The objective is ignored -- this trades
    # solution quality for runtime when all that's wanted is *a* feasible
    # collective algorithm under the given R.
    stop_at_feasible: bool = False

    # Iterative formulation strengthening - after the LP solve, check that the
    # tree decomposition recovers the full flow volume; if not, duplicate the
    # offending vertices in the topology and re-solve, up to strengthen_max_rounds
    # times.
    strengthen: bool = False
    strengthen_max_rounds: int = 3
    strengthen_tol: float = 1e-4

    # Rate-retry on infeasibility: if a DW solve's Phase 1 is infeasible (R below
    # the achievable rate), multiply R by rate_retry_factor and re-solve, up to
    # rate_retry_max times. rate_retry_max = 0 (default) disables this and keeps
    # the current behavior of terminating with an error. Applies to every DW
    # solve (single-collective, multi-collective, and strengthening rounds).
    rate_retry_max: int = 0
    rate_retry_factor: float = 1.01

    # Algorithm settings
    time_steps: int = 4
    dw_tolerance: float = 1e-7  # Dantzig-Wolfe convergence threshold
    all_reduce_via_reflection: bool = True
    n_workers: int = -1  # -1 = use mp.cpu_count()

    # Topology settings
    multirail_bandwidth: float = 25.0  # GBps, used when num_nodes > 1
    topology_cache_dir: str = "./topology_cache"

    # Output options
    step_schedule_K: int = 20
    total_data_size: int = 100000000  # Size of data in kB
    decomposition_tolerance: float = 1e-6
    uniform_chunks: bool = True
    uniform_chunk_size: int = 500  # Size of chunks in kB
    schedule_from_counts: bool = True
    # Expand the generated step schedule over every equivalence-class member. False (default)
    # emits only one representative origin per EC.
    expand_schedule_symmetry: bool = False


def _load_toml(p: Path) -> Config:
    with open(p, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise OptcclError(f"{p}: invalid TOML ({e})") from e
    return _parse(data)


def load_config(path: Optional[str] = None) -> Config:
    """Load config from --config path, fall back to ./optccl.toml
    Missing keys fall back to Config defaults."""
    if path:
        p = Path(path)
        if not p.exists():
            raise OptcclError(f"config file not found: {path}")
        return _load_toml(p)

    p = Path.cwd() / "optccl.toml"
    if p.exists():
        return _load_toml(p)
    return Config()


# Maps each TOML section to the Config fields it may set. Used both to route
# values in _parse and to warn about unknown sections/keys in the config file.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "gurobi": ("wls_access_id", "wls_secret", "license_id"),
    "solver": (
        "ag_no_crossover",
        "aa_no_crossover",
        "ar_no_crossover",
        "rs_no_crossover",
        "ag_phase1_cost_weight",
        "ar_phase1_cost_weight",
        "aa_phase1_cost_weight",
        "rs_phase1_cost_weight",
        "multi_phase1_cost_weight",
        "stop_at_feasible",
        "strengthen",
        "strengthen_max_rounds",
        "strengthen_tol",
        "rate_retry_max",
        "rate_retry_factor",
    ),
    "algorithm": (
        "time_steps",
        "dw_tolerance",
        "all_reduce_via_reflection",
        "n_workers",
    ),
    "topology": ("multirail_bandwidth", "topology_cache_dir"),
    "output": (
        "step_schedule_K",
        "total_data_size",
        "decomposition_tolerance",
        "uniform_chunks",
        "uniform_chunk_size",
        "schedule_from_counts",
        "expand_schedule_symmetry",
    ),
}


def _warn_unknown_keys(data: dict) -> None:
    for section, values in data.items():
        if section not in _SECTIONS:
            logger.warning("Unknown config section [%s] ignored", section)
            continue
        if not isinstance(values, dict):
            continue
        known = _SECTIONS[section]
        for key in values:
            if key not in known:
                logger.warning("Unknown config key '%s' in [%s] ignored", key, section)


def _parse(data: dict) -> Config:
    _warn_unknown_keys(data)
    cfg = Config()
    for section, keys in _SECTIONS.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            continue
        for key in keys:
            if key in values:
                setattr(cfg, key, values[key])
    return cfg


def _wls_configured(cfg: Config) -> bool:
    return all(v and v != "XXX" for v in (cfg.wls_access_id, cfg.wls_secret))


def make_gurobi_env(cfg: Config):
    import gurobipy as gp

    use_wls = _wls_configured(cfg)
    env = gp.Env(empty=True)
    if use_wls:
        env.setParam("WLSACCESSID", cfg.wls_access_id)
        env.setParam("WLSSECRET", cfg.wls_secret)
        env.setParam("LICENSEID", cfg.license_id)
    env.setParam("OutputFlag", 0)
    try:
        env.start()
    except gp.GurobiError as e:
        if use_wls:
            raise OptcclError(
                f"could not start the Gurobi WLS environment: {e}. Check the "
                f"[gurobi] credentials in your optccl.toml."
            ) from e
        raise OptcclError(
            f"could not start a Gurobi environment: {e}. optccl needs a Gurobi "
            f"license -- either install a local license (found automatically or "
            f"via GRB_LICENSE_FILE), or copy optccl.example.toml to "
            f"./optccl.toml and fill in your WLS credentials."
        ) from e
    return env
