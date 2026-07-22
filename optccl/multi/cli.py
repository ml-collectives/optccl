import json
import logging
import os

from ..config import load_config, make_gurobi_env
from .experiments import run_mode
from .workload import load_workload_spec
from .symmetry import validate_workload_ecs

logger = logging.getLogger(__name__)

MODES = ["concat", "overlay", "joint"]

_WORKLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "workloads",
)
WORKLOAD_ALIASES = {
    "nv": "a100_nv_joint.json",
    "pcie": "a100_pcie_joint.json",
    "both": "a100_both_joint.json",
}


def add_multi_parser(subparsers, common=None):
    parents = [common] if common is not None else []
    p_multi = subparsers.add_parser(
        "multi",
        help="solve multiple collectives together against shared bandwidth",
    )
    multi_sub = p_multi.add_subparsers(dest="multi_command", required=True)

    p_solve = multi_sub.add_parser(
        "solve",
        help="solve and process a multi-collective workload",
        parents=parents,
    )
    p_solve.add_argument(
        "workload",
        metavar="WORKLOAD",
        help="path to an optccl-workload-spec JSON, or one of the bundled paper "
        f"workloads: {', '.join(WORKLOAD_ALIASES)} (the A100 AG+AA experiments)",
    )
    p_solve.add_argument(
        "--mode",
        choices=MODES,
        default="joint",
        help="how to combine the collectives: concat (concat_groups run sequentially), "
        "overlay (groups solved separately, run simultaneously), joint (one combined "
        "LP, families decomposed on residual bandwidth). Default joint.",
    )
    p_solve.add_argument(
        "-r",
        "-R",
        "--rval",
        type=float,
        default=None,
        help="target rate R for joint mode: the schedule length per unit of data (the "
        "INVERSE of throughput; auto-determined if omitted -- pass a LARGER value "
        "to relax an infeasible solve). Ignored by concat/overlay, which "
        "auto-determine each group's own rate.",
    )
    p_solve.add_argument(
        "-T",
        "--time-steps",
        type=int,
        default=None,
        metavar="N",
        help="number of steps in the time-expanded LP (overrides the config's time_steps)",
    )
    p_solve.add_argument(
        "--validate-symmetry",
        action="store_true",
        help="verify the workload's demand-aware equivalence classes (partition, "
        "transversal, every shift a demand-preserving automorphism) before solving",
    )
    p_solve.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="write result JSON (overhead/latency breakdown) to this file",
    )
    p_solve.set_defaults(func=cmd_multi_solve)
    return p_multi


def _fmt(x):
    return "    n/a" if x is None else f"{x:>10.4f}"


def _resolve_workload_path(arg: str) -> str:
    if arg in WORKLOAD_ALIASES:
        return os.path.join(_WORKLOADS_DIR, WORKLOAD_ALIASES[arg])
    return arg


def cmd_multi_solve(args):
    cfg = load_config(args.config)
    if args.time_steps is not None:
        cfg.time_steps = args.time_steps
    gurobi_env = make_gurobi_env(cfg)

    wl = load_workload_spec(_resolve_workload_path(args.workload), cfg)
    logger.info(
        "Workload '%s': %d collectives -> %d p2p / %d gather / %d reduce "
        "demands on %d GPUs; ECs: %s",
        wl.name,
        len(wl.collectives),
        len(wl.p2ps),
        len(wl.gathers),
        len(wl.reduces),
        len(wl.effective_gpus),
        sorted((len(ec.gpus) for ec in wl.ECs), reverse=True),
    )
    if args.validate_symmetry:
        validate_workload_ecs(wl)
        logger.info("Demand-aware ECs validated.")

    result = run_mode(args.mode, wl, gurobi_env, cfg, R=args.rval)

    logger.info("multi %s on '%s'  (scale=%g):", args.mode, wl.name, result["scale"])
    logger.info("  %-10s %10s %10s", "", "ideal", "realized")
    for label, vals in (result.get("breakdown") or {}).items():
        logger.info(
            "  %-10s %s %s", label, _fmt(vals.get("ideal")), _fmt(vals.get("realized"))
        )
    logger.info(
        "  %-10s %s %s",
        "COMBINED",
        _fmt(result["ideal_latency"]),
        _fmt(result["realized_latency"]),
    )

    for k, vv in result.get("components", {}).items():
        if isinstance(vv, float):
            logger.info("  %-11s: %.6f", k, vv)
        else:
            logger.info("  %-11s: %s", k, vv)
    logger.info("  solve time: %.2fs", result["solve_time"])

    if args.output:
        out = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        logger.info("Result written to %s", args.output)
