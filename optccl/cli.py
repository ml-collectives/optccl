import argparse
import json
import logging
import os
import re
import sys

from . import __version__
from .topology_spec import TOPOLOGIES, load_or_build_topology
from .config import load_config, make_gurobi_env
from .errors import OptcclError
from .logging_setup import configure_logging
from .solver import (
    all_gather_lp,
    reduce_scatter_lp,
    all_reduce_lp,
    all_to_all_lp,
    postprocess,
)
from .serialize import save_solve_result, load_solve_result
from .multi.cli import add_multi_parser

logger = logging.getLogger(__name__)

COLLECTIVES = ["all_gather", "reduce_scatter", "all_reduce", "all_to_all"]


def _topology_slug(topology_name: str) -> str:
    stem = os.path.splitext(os.path.basename(topology_name))[0]
    return re.sub(r"[^A-Za-z0-9.-]+", "_", stem).strip("_") or "topology"


def _auto_output(prefix: str, *parts, directory: str = "") -> str:
    path = os.path.join(
        directory, "_".join([prefix, *(str(p) for p in parts)]) + ".json"
    )
    logger.info("No -o/--output given; using %s", path)
    return path


def _add_rate_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-r",
        "-R",
        "--rval",
        type=float,
        default=None,
        help="target rate R: the schedule length per unit of data "
        "(the INVERSE of throughput; auto-determined from a cut "
        "bound if omitted). If a solve reports R below the "
        "achievable rate, pass a LARGER value.",
    )
    parser.add_argument(
        "-T",
        "--time-steps",
        type=int,
        default=None,
        metavar="N",
        help="number of steps in the time-expanded LP "
        "(overrides the config's time_steps)",
    )


def _add_solve_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "topology",
        metavar="TOPOLOGY",
        help=f"built-in topology name ({', '.join(TOPOLOGIES)}) "
        f"or a path to a .json topology spec",
    )
    parser.add_argument("num_nodes", type=int, help="number of nodes")
    _add_rate_args(parser)
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="path to write the solver result JSON (default: "
        "sol_<collective>_<topology>_<num_nodes>.json)",
    )
    parser.add_argument(
        "--strengthen",
        action="store_true",
        help="iteratively strengthen the formulation (duplicate offending "
        "vertices and re-solve) until the tree decomposition recovers "
        "the full flow volume (all_gather, reduce_scatter, all_reduce)",
    )


def _add_run_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "topology",
        metavar="TOPOLOGY",
        help=f"built-in topology name ({', '.join(TOPOLOGIES)}) "
        f"or a path to a .json topology spec",
    )
    parser.add_argument("num_nodes", type=int, help="number of nodes")
    _add_rate_args(parser)
    parser.add_argument(
        "--collective",
        metavar="COLLECTIVE",
        default=None,
        help="override collective for schedule interpretation",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="write the result JSON (overhead/latency analysis; with "
        "--write-schedule, also the full step schedule)",
    )
    parser.add_argument(
        "--write-schedule",
        action="store_true",
        help="run the exact WFQ and include the full step schedule in -o "
        "(auto-named process_<collective>_<topology>_<num_nodes>.json if "
        "-o is omitted). Without it, run reports overhead via the "
        "count-based sim without materializing a schedule.",
    )
    parser.add_argument(
        "--strengthen",
        action="store_true",
        help="iteratively strengthen the formulation (duplicate offending "
        "vertices and re-solve) until the tree decomposition recovers "
        "the full flow volume (all_gather, reduce_scatter, all_reduce)",
    )


def _flows_from_lp_result(result: dict, solver_type: str) -> dict:
    if solver_type == "tree":
        return {"f": result["f"], "w": result["w"]}
    elif solver_type == "tree_paired":
        return {
            "f1": result["f1"],
            "w1": result["w1"],
            "f2": result["f2"],
            "w2": result["w2"],
        }
    elif solver_type == "path":
        return {"f": result["f"]}
    raise ValueError(f"Unknown solver_type {solver_type!r}")


def _solve(collective: str, lp_fn, args, cfg, gurobi_env):
    logger.info(
        "solving %s (%s, %d node(s))...", collective, args.topology, args.num_nodes
    )
    if getattr(args, "strengthen", False):
        cfg.strengthen = True
    top = load_or_build_topology(args.topology, args.num_nodes, cfg)
    result = lp_fn(top, args.rval, cfg, gurobi_env=gurobi_env)
    # A strengthened solve returns the enlarged topology its flows live on; all
    # downstream decomposition/scheduling must run against that topology.
    top = result.get("topology", top)
    solver_type = result["solver_type"]
    metadata = {
        "collective": collective,
        "solver_type": solver_type,
        "topology_name": args.topology,
        "num_nodes": args.num_nodes,
        "T": result["T"],
        "lp_rval": result["lp_rval"],
        "rval": result["rval"],
        "objective_value": result["objective_value"],
        "solve_time": result["solve_time"],
    }
    if "strengthen_rounds" in result:
        metadata["strengthen_rounds"] = result["strengthen_rounds"]
    flows = _flows_from_lp_result(result, solver_type)
    return metadata, top, flows


def _apply_cli_overrides(cfg, args):
    if getattr(args, "time_steps", None) is not None:
        cfg.time_steps = args.time_steps


def _run_solve(collective: str, lp_fn, args):
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    gurobi_env = make_gurobi_env(cfg)
    output = args.output or _auto_output(
        "sol", collective, _topology_slug(args.topology), args.num_nodes
    )
    metadata, top, flows = _solve(collective, lp_fn, args, cfg, gurobi_env)
    save_solve_result(output, metadata, top, flows)
    logger.info("  objective value : %.6f", metadata["objective_value"])
    logger.info("  solve time      : %.2fs", metadata["solve_time"])


def _validate_process_output_args(collective_override):
    if collective_override and collective_override not in COLLECTIVES:
        raise OptcclError(
            f"unknown collective '{collective_override}'. Available: {', '.join(COLLECTIVES)}"
        )


def _process_and_report(
    metadata, top, flows, cfg, collective_override, write_schedule, output, gurobi_env
):
    result = postprocess(
        metadata,
        top,
        flows,
        cfg,
        collective=collective_override,
        write_schedule=write_schedule,
        gurobi_env=gurobi_env,
    )

    schedule = result.get("schedule")
    if write_schedule:
        if schedule is not None:
            coverage = (
                "all equivalence-class members"
                if result.get("schedule_expanded")
                else "one representative per EC; expand via 'ec_expansion'"
            )
            logger.info("Schedule generated: %d step(s) (%s)", len(schedule), coverage)
        with open(output, "w") as fh:
            json.dump(result, fh, default=_json_default)
        logger.info("Schedule written to %s", output)
    else:
        logger.info(
            "Overhead computed via the count-based sim (%d steps); "
            "no schedule generated (pass --write-schedule to include one).",
            result.get("n_steps", 0),
        )
        if output:
            with open(output, "w") as fh:
                json.dump(result, fh, default=_json_default)
            logger.info("Overhead/latency analysis written to %s", output)

    logger.info("  objective value : %.6f", result["objective_value"])
    logger.info("  original solve  : %.2fs", result["solve_time"])
    if "overhead" in result:
        logger.info(
            "  overhead        : %.1f%%  (variable-step; realized %.2f vs ideal %.2f)",
            result["overhead"] * 100,
            result["realized_latency"],
            result["ideal_latency"],
        )
        logger.info(
            "  overhead (unif) : %.1f%%  (uniform-step / barrier-per-step, for reference)",
            result["overhead_uniform_step"] * 100,
        )
    if "measured_latency_s" in result:
        logger.info(
            "  measured latency: %.4f s  (sum of variable step lengths)",
            result["measured_latency_s"],
        )
        logger.info(
            "  algorithmic bw  : %.2f GB/s  (num_gpus * data_size / latency)",
            result["algorithmic_bandwidth_GBps"],
        )


def _run_run(collective: str, lp_fn, args):
    _validate_process_output_args(args.collective)
    output = args.output
    if args.write_schedule and not output:
        output = _auto_output(
            "process",
            args.collective or collective,
            _topology_slug(args.topology),
            args.num_nodes,
        )
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    gurobi_env = make_gurobi_env(cfg)
    metadata, top, flows = _solve(collective, lp_fn, args, cfg, gurobi_env)
    _process_and_report(
        metadata,
        top,
        flows,
        cfg,
        args.collective,
        args.write_schedule,
        output,
        gurobi_env,
    )


def cmd_solve_all_gather(args):
    _run_solve("all_gather", all_gather_lp, args)


def cmd_solve_reduce_scatter(args):
    _run_solve("reduce_scatter", reduce_scatter_lp, args)


def cmd_solve_all_reduce(args):
    _run_solve("all_reduce", all_reduce_lp, args)


def cmd_solve_all_to_all(args):
    _run_solve("all_to_all", all_to_all_lp, args)


def cmd_run_all_gather(args):
    _run_run("all_gather", all_gather_lp, args)


def cmd_run_reduce_scatter(args):
    _run_run("reduce_scatter", reduce_scatter_lp, args)


def cmd_run_all_reduce(args):
    _run_run("all_reduce", all_reduce_lp, args)


def cmd_run_all_to_all(args):
    _run_run("all_to_all", all_to_all_lp, args)


def cmd_process(args):
    _validate_process_output_args(args.collective)
    output = args.output
    if args.write_schedule and not output:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        stem = stem[len("sol_") :] if stem.startswith("sol_") else stem
        output = _auto_output("process", stem, directory=os.path.dirname(args.input))
    metadata, top, flows = load_solve_result(args.input)
    cfg = load_config(args.config)
    gurobi_env = make_gurobi_env(cfg)
    _process_and_report(
        metadata,
        top,
        flows,
        cfg,
        args.collective,
        args.write_schedule,
        output,
        gurobi_env,
    )


def _json_default(obj):
    if isinstance(obj, (tuple, set)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config", metavar="PATH", help="path to optccl.toml config file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v shows per-iteration solver detail)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only log warnings and errors"
    )
    parser.add_argument(
        "--log-file", metavar="PATH", default=None, help="also write logs to this file"
    )


def main():
    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common)

    parser = argparse.ArgumentParser(
        prog="optccl",
        description="Optimal collective communication algorithm solver",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve subcommand group
    p_solve = subparsers.add_parser(
        "solve", help="run the LP solver and save results to file"
    )
    solve_sub = p_solve.add_subparsers(dest="collective", required=True)

    p_ag = solve_sub.add_parser(
        "all_gather", help="solve AllGather LP", parents=[common]
    )
    _add_solve_args(p_ag)
    p_ag.set_defaults(func=cmd_solve_all_gather)

    p_rs = solve_sub.add_parser(
        "reduce_scatter", help="solve ReduceScatter LP", parents=[common]
    )
    _add_solve_args(p_rs)
    p_rs.set_defaults(func=cmd_solve_reduce_scatter)

    p_ar = solve_sub.add_parser(
        "all_reduce", help="solve AllReduce LP", parents=[common]
    )
    _add_solve_args(p_ar)
    p_ar.set_defaults(func=cmd_solve_all_reduce)

    p_aa = solve_sub.add_parser(
        "all_to_all", help="solve All-to-All LP", parents=[common]
    )
    _add_solve_args(p_aa)
    p_aa.set_defaults(func=cmd_solve_all_to_all)

    # run subcommand group (solve + process, no intermediate file)
    p_run = subparsers.add_parser(
        "run",
        help="solve the LP and immediately post-process it (like `solve` then "
        "`process`, without writing the intermediate solver result to disk)",
    )
    run_sub = p_run.add_subparsers(dest="collective", required=True)

    p_run_ag = run_sub.add_parser(
        "all_gather", help="solve+process AllGather", parents=[common]
    )
    _add_run_args(p_run_ag)
    p_run_ag.set_defaults(func=cmd_run_all_gather)

    p_run_rs = run_sub.add_parser(
        "reduce_scatter", help="solve+process ReduceScatter", parents=[common]
    )
    _add_run_args(p_run_rs)
    p_run_rs.set_defaults(func=cmd_run_reduce_scatter)

    p_run_ar = run_sub.add_parser(
        "all_reduce", help="solve+process AllReduce", parents=[common]
    )
    _add_run_args(p_run_ar)
    p_run_ar.set_defaults(func=cmd_run_all_reduce)

    p_run_aa = run_sub.add_parser(
        "all_to_all", help="solve+process All-to-All", parents=[common]
    )
    _add_run_args(p_run_aa)
    p_run_aa.set_defaults(func=cmd_run_all_to_all)

    # process command
    p_proc = subparsers.add_parser(
        "process",
        help="run post-processing (tree decomposition + scheduling) on a saved solver result",
        parents=[common],
    )
    p_proc.add_argument(
        "input", metavar="FILE", help="solver result JSON produced by `optccl solve`"
    )
    p_proc.add_argument(
        "--collective",
        metavar="COLLECTIVE",
        default=None,
        help="override collective for schedule interpretation",
    )
    p_proc.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="write the result JSON (overhead/latency analysis; with "
        "--write-schedule, also the full step schedule)",
    )
    p_proc.add_argument(
        "--write-schedule",
        action="store_true",
        help="run the exact WFQ and include the full step schedule in -o "
        "(auto-named process_<input-stem>.json next to the input if -o is "
        "omitted). Without it, process reports overhead via the "
        "count-based sim without materializing a schedule.",
    )
    p_proc.set_defaults(func=cmd_process)

    add_multi_parser(subparsers, common)

    args = parser.parse_args()
    configure_logging(args.verbose, args.quiet, args.log_file)
    try:
        args.func(args)
    except OptcclError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
