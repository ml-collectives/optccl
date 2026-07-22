import logging

from .topologies import Topology, validate_ecs
from .config import Config, make_gurobi_env
from .strengthen import (
    StrengthenState,
    apply_strengthening,
    check_full_volume,
    detect_offending,
    merge_strengthened_topologies,
    refine_state,
)
from .tree_decomposer import decompose_trees, reverse_time_expanded
from .wfq import (
    wfq_schedule,
    wfq_schedule_from_counts,
    wfq_path_schedule,
    wfq_allreduce_schedule,
    wfq_allreduce_schedule_from_counts,
)

from . import all_gather as ag
from . import all_to_all as aa
from . import all_reduce as ar
from . import reduce_scatter as rs
from .process_results import (
    post_process_trees,
    decompose_and_process_paths,
    collect_wfq_scheds,
    collect_raw_tree_scheds,
    collect_wfq_path_scheds,
    collect_raw_path_scheds,
    collect_allreduce_wfq_scheds,
    collect_raw_allreduce_scheds,
    interpret_ts_as_all_gather,
    interpret_ts_as_reduce_scatter,
    interpret_ts_as_all_reduce,
    interpret_ps_as_all_to_all,
    interpret_tts_as_all_reduce,
    analyze_step_loads,
    format_step_load_report,
    count_tree_overhead,
    count_path_overhead,
    count_allreduce_overhead,
    ec_expansion_metadata,
)

logger = logging.getLogger(__name__)


def _decompose_to_wfq_trees(
    collective, top, g, f, w, R, gurobi_env, cfg, full_budget=True
):
    decomposed = decompose_trees(
        top, g, f, w, collective, gurobi_env, cfg, R, full_budget=full_budget
    )
    # Pass-through nodes are those without storage; they get elided
    # from the logical tree during post-processing.
    switches = set(top.components) - top.storage_nodes
    return post_process_trees(decomposed, switches)


def _allreduce_trees(top, g, f1, w1, f2, w2, R, gurobi_env, cfg):
    T = max(node[1] for node in g.nodes())
    ag_trees = _decompose_to_wfq_trees(
        "all_gather", top, g, f1, w1, R, gurobi_env, cfg, full_budget=False
    )
    g_rev, f2_rev, w2_rev = reverse_time_expanded(g, f2, w2, T)
    # A branch in the reversed out-tree is a reduction in real time, so the
    # reversed RS flow may only branch at reduce-capable nodes.
    rs_trees = _decompose_to_wfq_trees(
        "reduce_scatter",
        top,
        g_rev,
        f2_rev,
        w2_rev,
        R,
        gurobi_env,
        cfg,
        full_budget=False,
    )
    return rs_trees, ag_trees


def _allreduce_paired_schedule(
    top, g, f1, w1, f2, w2, R, gurobi_env, cfg, write_schedule=False
):
    rs_trees, ag_trees = _allreduce_trees(top, g, f1, w1, f2, w2, R, gurobi_env, cfg)
    C = cfg.uniform_chunk_size
    K = cfg.step_schedule_K
    D = cfg.total_data_size

    if not write_schedule:
        analysis = count_allreduce_overhead(rs_trees, ag_trees, top, D, C, K, R)
        return None, analysis

    expand = cfg.expand_schedule_symmetry
    if cfg.uniform_chunks:
        gen = (
            wfq_allreduce_schedule_from_counts
            if cfg.schedule_from_counts
            else wfq_allreduce_schedule
        )
        rs_scheds, ag_scheds = {}, {}
        for ec in top.ECs:
            o = ec.gpus[0]
            rs_s, ag_s = gen(rs_trees[o], ag_trees[o], D, C, K, R)
            rs_scheds[o] = rs_s
            ag_scheds[o] = ag_s
        sched = collect_allreduce_wfq_scheds(
            rs_scheds, ag_scheds, C, top.ECs, expand=expand
        )
    else:
        sched = collect_raw_allreduce_scheds(
            rs_trees,
            ag_trees,
            cfg.total_data_size,
            cfg.step_schedule_K,
            top.ECs,
            expand=expand,
        )
    analysis = (
        analyze_step_loads(sched, top)
        if expand
        else count_allreduce_overhead(rs_trees, ag_trees, top, D, C, K, R)
    )
    return sched, analysis


def _tree_schedule(collective, top, g, f, w, R, gurobi_env, cfg, write_schedule=False):
    decomposed_trees = decompose_trees(top, g, f, w, collective, gurobi_env, cfg, R)
    switches = set(top.components) - top.storage_nodes
    wfq_input_trees = post_process_trees(decomposed_trees, switches)

    if not write_schedule:
        analysis = count_tree_overhead(
            wfq_input_trees,
            top,
            cfg.total_data_size,
            cfg.uniform_chunk_size,
            cfg.step_schedule_K,
            R,
        )
        return None, analysis

    expand = cfg.expand_schedule_symmetry
    if cfg.uniform_chunks:
        gen = wfq_schedule_from_counts if cfg.schedule_from_counts else wfq_schedule
        scheds = {}
        for ec in top.ECs:
            o = ec.gpus[0]
            scheds[o] = gen(
                wfq_input_trees[o],
                cfg.total_data_size,
                cfg.uniform_chunk_size,
                cfg.step_schedule_K,
                R,
                required_chunks=cfg.total_data_size / cfg.uniform_chunk_size,
            )
        sched = collect_wfq_scheds(
            scheds, cfg.uniform_chunk_size, top.ECs, expand=expand
        )
    else:
        sched = collect_raw_tree_scheds(
            wfq_input_trees,
            cfg.total_data_size,
            cfg.step_schedule_K,
            top.ECs,
            expand=expand,
        )
    analysis = (
        analyze_step_loads(sched, top)
        if expand
        else count_tree_overhead(
            wfq_input_trees,
            top,
            cfg.total_data_size,
            cfg.uniform_chunk_size,
            cfg.step_schedule_K,
            R,
        )
    )
    return sched, analysis


def _interpret_tree_schedule(ts, collective):
    if collective == "all_gather":
        return interpret_ts_as_all_gather(ts)
    elif collective == "all_reduce":
        return interpret_ts_as_all_reduce(ts)
    elif collective == "reduce_scatter":
        return interpret_ts_as_reduce_scatter(ts)
    else:
        logger.warning(
            "Unknown collective %r for tree schedule, defaulting to all_gather",
            collective,
        )
        return interpret_ts_as_all_gather(ts)


def _path_schedule(top, g, f, R, cfg, write_schedule=False):
    wfq_input_paths = decompose_and_process_paths(top, g, f, cfg)

    if not write_schedule:
        analysis = count_path_overhead(
            wfq_input_paths,
            top,
            cfg.total_data_size,
            cfg.uniform_chunk_size,
            cfg.step_schedule_K,
            R,
        )
        return None, analysis

    expand = cfg.expand_schedule_symmetry
    if cfg.uniform_chunks:
        scheds = {}
        for ec in top.ECs:
            o = ec.gpus[0]
            if cfg.schedule_from_counts:
                scheds[o] = wfq_schedule_from_counts(
                    wfq_input_paths[o],
                    cfg.total_data_size,
                    cfg.uniform_chunk_size,
                    cfg.step_schedule_K,
                    R,
                    path_mode=True,
                )
            else:
                scheds[o] = wfq_path_schedule(
                    wfq_input_paths[o],
                    cfg.total_data_size,
                    cfg.uniform_chunk_size,
                    cfg.step_schedule_K,
                    R,
                    required_chunks=None,
                )
        sched = collect_wfq_path_scheds(
            scheds, cfg.uniform_chunk_size, top.ECs, expand=expand
        )
    else:
        sched = collect_raw_path_scheds(
            wfq_input_paths,
            cfg.total_data_size,
            cfg.step_schedule_K,
            top.ECs,
            expand=expand,
        )
    analysis = (
        analyze_step_loads(sched, top)
        if expand
        else count_path_overhead(
            wfq_input_paths,
            top,
            cfg.total_data_size,
            cfg.uniform_chunk_size,
            cfg.step_schedule_K,
            R,
        )
    )
    return sched, analysis


def all_gather_lp(top: Topology, R=None, cfg: Config = None, gurobi_env=None):
    if cfg is None:
        cfg = Config()
    if gurobi_env is None:
        gurobi_env = make_gurobi_env(cfg)
    if R is None:
        R = ag.determine_throughput(top, gurobi_env, reverse=True)
    if cfg.strengthen:
        return _strengthened_all_gather_lp(top, R, cfg, gurobi_env)
    m, f, w, g, solve_time, R = ag.all_gather_mirrored_dw(
        top, cfg.time_steps, R, gurobi_env, cfg
    )
    return {
        "solver_type": "tree",
        "T": cfg.time_steps,
        "lp_rval": R,
        "rval": R,
        "objective_value": m.ObjVal,
        "solve_time": solve_time,
        "f": f,
        "w": w,
        "g": g,
    }


def _run_strengthen_loop(
    base_top: Topology, cfg: Config, gurobi_env, run_round, allowed_kinds=None
):
    state = StrengthenState()
    top = base_top
    total_time = 0.0
    rounds = 0
    for rnd in range(cfg.strengthen_max_rounds + 1):
        bundle, checks = run_round(top)
        total_time += bundle["solve_time"]
        shortfall = {}
        for decomposed, _w, _g, _orient in checks:
            for o, obj in check_full_volume(decomposed, cfg.strengthen_tol).items():
                shortfall[o] = min(obj, shortfall.get(o, 1.0))
        if not shortfall:
            if rnd:
                logger.info(
                    "strengthening converged after %d round(s) (%s)",
                    rnd,
                    state.describe(),
                )
            else:
                logger.info(
                    "strengthening: decomposition already recovers the "
                    "full volume; no duplication needed"
                )
            break
        if rnd == cfg.strengthen_max_rounds:
            logger.warning(
                "strengthening: reached max rounds (%d) with origins still below "
                "full volume: %s; returning the best solution found",
                cfg.strengthen_max_rounds,
                {o: round(v, 4) for o, v in shortfall.items()},
            )
            break
        logger.info(
            "strengthening round %d: origins below full volume: %s",
            rnd + 1,
            {o: round(v, 4) for o, v in shortfall.items()},
        )
        proposals = []
        for decomposed, w_used, g_used, orient in checks:
            proposals += detect_offending(
                decomposed,
                w_used,
                g_used,
                base_top,
                cfg,
                gurobi_env,
                orientation=orient,
            )
        if allowed_kinds is not None:
            dropped = [p for p in proposals if p[0] not in allowed_kinds]
            if dropped:
                logger.warning(
                    "strengthening: dropping %d proposal(s) of kind(s) %s -- "
                    "unsound for a formulation sharing both flow families; "
                    "only switch duplication applies here",
                    len(dropped),
                    sorted({p[0] for p in dropped}),
                )
            proposals = [p for p in proposals if p[0] in allowed_kinds]
        if not refine_state(state, proposals, base_top):
            logger.warning(
                "strengthening: no further vertex duplication is possible; "
                "returning the best solution found (origins below full volume: %s)",
                {o: round(v, 4) for o, v in shortfall.items()},
            )
            break
        top = apply_strengthening(base_top, state, cfg.time_steps)
        validate_ecs(top)
        rounds = rnd + 1
        logger.info(
            "strengthening: re-solving on enlarged topology (%s; %d -> %d components)",
            state.describe(),
            len(base_top.components),
            len(top.components),
        )
    bundle["solve_time"] = total_time
    bundle["topology"] = top
    bundle["strengthen_rounds"] = rounds
    return bundle


def _strengthened_all_gather_lp(
    top: Topology, R, cfg: Config, gurobi_env, full_budget=True
):
    def run_round(top_k):
        m, f, w, g, st, R_eff = ag.all_gather_mirrored_dw(
            top_k, cfg.time_steps, R, gurobi_env, cfg
        )
        decomposed = decompose_trees(
            top_k,
            g,
            f,
            w,
            "all_gather",
            gurobi_env,
            cfg,
            R_eff,
            full_budget=full_budget,
        )
        bundle = {
            "solver_type": "tree",
            "T": cfg.time_steps,
            "lp_rval": R_eff,
            "rval": R_eff,
            "objective_value": m.ObjVal,
            "solve_time": st,
            "f": f,
            "w": w,
            "g": g,
        }
        return bundle, [(decomposed, w, g, "fanout")]

    return _run_strengthen_loop(top, cfg, gurobi_env, run_round)


def _strengthened_reduce_scatter_lp(
    top: Topology, R, cfg: Config, gurobi_env, full_budget=True
):
    def run_round(top_k):
        m, f, w, g, st, R_eff = rs.reduce_scatter_mirrored_dw(
            top_k, cfg.time_steps, R, gurobi_env, cfg
        )
        # The RS flow is a fan-in; the decomposition (and hence the
        # feasibility check and offending-edge detection) lives in the
        # time-reversed broadcast orientation.
        g_rev, f_rev, w_rev = reverse_time_expanded(g, f, w, cfg.time_steps)
        decomposed = decompose_trees(
            top_k,
            g_rev,
            f_rev,
            w_rev,
            "reduce_scatter",
            gurobi_env,
            cfg,
            R_eff,
            full_budget=full_budget,
        )
        bundle = {
            "solver_type": "tree",
            "T": cfg.time_steps,
            "lp_rval": R_eff,
            "rval": R_eff,
            "objective_value": m.ObjVal,
            "solve_time": st,
            "f": f,
            "w": w,
            "g": g,
        }
        return bundle, [(decomposed, w_rev, g_rev, "fanin")]

    return _run_strengthen_loop(top, cfg, gurobi_env, run_round)


def _strengthened_all_reduce_paired_lp(top: Topology, R, cfg: Config, gurobi_env):
    def run_round(top_k):
        m, f1, w1, f2, w2, g, st, R_eff = ar.all_reduce_mirrored_dw(
            top_k, cfg.time_steps, R, gurobi_env, cfg
        )
        dec_ag = decompose_trees(
            top_k, g, f1, w1, "all_gather", gurobi_env, cfg, R_eff, full_budget=False
        )
        g_rev, f2_rev, w2_rev = reverse_time_expanded(g, f2, w2, cfg.time_steps)
        dec_rs = decompose_trees(
            top_k,
            g_rev,
            f2_rev,
            w2_rev,
            "reduce_scatter",
            gurobi_env,
            cfg,
            R_eff,
            full_budget=False,
        )
        bundle = {
            "solver_type": "tree_paired",
            "T": cfg.time_steps,
            "lp_rval": R_eff,
            "rval": R_eff,
            "objective_value": m.ObjVal,
            "solve_time": st,
            "f1": f1,
            "w1": w1,
            "f2": f2,
            "w2": w2,
            "g": g,
        }
        return bundle, [(dec_ag, w1, g, "fanout"), (dec_rs, w2_rev, g_rev, "fanin")]

    return _run_strengthen_loop(
        top, cfg, gurobi_env, run_round, allowed_kinds={"vc", "vcr"}
    )


def reduce_scatter_lp(top: Topology, R=None, cfg: Config = None, gurobi_env=None):
    if cfg is None:
        cfg = Config()
    if gurobi_env is None:
        gurobi_env = make_gurobi_env(cfg)
    if R is None:
        R = ag.determine_throughput(top, gurobi_env)
    if cfg.strengthen:
        return _strengthened_reduce_scatter_lp(top, R, cfg, gurobi_env)
    m, f, w, g, solve_time, R = rs.reduce_scatter_mirrored_dw(
        top, cfg.time_steps, R, gurobi_env, cfg
    )
    return {
        "solver_type": "tree",
        "T": cfg.time_steps,
        "lp_rval": R,
        "rval": R,
        "objective_value": m.ObjVal,
        "solve_time": solve_time,
        "f": f,
        "w": w,
        "g": g,
    }


def all_reduce_lp(top: Topology, R=None, cfg: Config = None, gurobi_env=None):
    """Run the all_reduce LP and return raw results."""
    if cfg is None:
        cfg = Config()
    if gurobi_env is None:
        gurobi_env = make_gurobi_env(cfg)
    if cfg.all_reduce_via_reflection:
        if top.is_self_reflective():
            lp_R = ag.determine_throughput(top, gurobi_env) if R is None else R / 2
            if cfg.strengthen:
                res = _strengthened_all_gather_lp(top, lp_R, cfg, gurobi_env)
                res["rval"] = res["lp_rval"] * 2
                res["objective_value"] = 2 * res["objective_value"]
                return res
            m, f, w, g, solve_time, lp_R = ag.all_gather_mirrored_dw(
                top, cfg.time_steps, lp_R, gurobi_env, cfg
            )
            return {
                "solver_type": "tree",
                "T": cfg.time_steps,
                "lp_rval": lp_R,
                "rval": lp_R * 2,
                "objective_value": 2 * m.ObjVal,
                "solve_time": solve_time,
                "f": f,
                "w": w,
                "g": g,
            }
        logger.info(
            "Topology is not self-reflective; solving separate all_gather and "
            "reduce_scatter LPs instead of reflecting the all_gather solution."
        )
        if R is None:
            R_ag = ag.determine_throughput(top, gurobi_env, reverse=True)
            R_rs = ag.determine_throughput(top, gurobi_env)
        else:
            R_ag = R_rs = R / 2
        if cfg.strengthen:
            res_ag = _strengthened_all_gather_lp(
                top, R_ag, cfg, gurobi_env, full_budget=False
            )
            res_rs = _strengthened_reduce_scatter_lp(
                top, R_rs, cfg, gurobi_env, full_budget=False
            )
            # Each phase's effective rate (possibly bumped by rate-retry).
            R_ag, R_rs = res_ag["lp_rval"], res_rs["lp_rval"]
            merged = merge_strengthened_topologies(
                top, [res_ag["topology"], res_rs["topology"]]
            )
            validate_ecs(merged)
            return {
                "solver_type": "tree_paired",
                "T": cfg.time_steps,
                "lp_rval": R_ag + R_rs,
                "rval": R_ag + R_rs,
                "lp_rval_ag": R_ag,
                "lp_rval_rs": R_rs,
                "objective_value": res_ag["objective_value"]
                + res_rs["objective_value"],
                "solve_time": res_ag["solve_time"] + res_rs["solve_time"],
                "f1": res_ag["f"],
                "w1": res_ag["w"],
                "f2": res_rs["f"],
                "w2": res_rs["w"],
                "g": res_ag["g"],
                "topology": merged,
                "strengthen_rounds": res_ag["strengthen_rounds"]
                + res_rs["strengthen_rounds"],
            }
        m1, f1, w1, g, st1, R_ag = ag.all_gather_mirrored_dw(
            top, cfg.time_steps, R_ag, gurobi_env, cfg
        )
        m2, f2, w2, _, st2, R_rs = rs.reduce_scatter_mirrored_dw(
            top, cfg.time_steps, R_rs, gurobi_env, cfg
        )
        return {
            "solver_type": "tree_paired",
            "T": cfg.time_steps,
            "lp_rval": R_ag + R_rs,
            "rval": R_ag + R_rs,
            "lp_rval_ag": R_ag,
            "lp_rval_rs": R_rs,
            "objective_value": m1.ObjVal + m2.ObjVal,
            "solve_time": st1 + st2,
            "f1": f1,
            "w1": w1,
            "f2": f2,
            "w2": w2,
            "g": g,
        }
    else:
        if R is None:
            # The paired LP runs both phases against shared bandwidth, so its
            # budget is the sum of the per-phase bounds.
            R = ag.determine_throughput(
                top, gurobi_env, reverse=True
            ) + ag.determine_throughput(top, gurobi_env)
        if cfg.strengthen:
            return _strengthened_all_reduce_paired_lp(top, R, cfg, gurobi_env)
        m, f1, w1, f2, w2, g, solve_time, R = ar.all_reduce_mirrored_dw(
            top, cfg.time_steps, R, gurobi_env, cfg
        )
        return {
            "solver_type": "tree_paired",
            "T": cfg.time_steps,
            "lp_rval": R,
            "rval": R,
            "objective_value": m.ObjVal,
            "solve_time": solve_time,
            "f1": f1,
            "w1": w1,
            "f2": f2,
            "w2": w2,
            "g": g,
        }


def all_to_all_lp(top: Topology, R=None, cfg: Config = None, gurobi_env=None):
    if cfg is None:
        cfg = Config()
    if gurobi_env is None:
        gurobi_env = make_gurobi_env(cfg)
    if R is None:
        R = aa.determine_throughput(top, gurobi_env)
    if cfg.strengthen:
        logger.info(
            "strengthening does not apply to all_to_all (no w envelope; "
            "each flow is priced directly, so usage merging cannot occur)"
        )
    m, f, g, solve_time, R = aa.all_to_all_mirrored_dw(
        top, cfg.time_steps, R, gurobi_env, cfg
    )
    return {
        "solver_type": "path",
        "T": cfg.time_steps,
        "lp_rval": R,
        "rval": R,
        "objective_value": m.ObjVal,
        "solve_time": solve_time,
        "f": f,
        "g": g,
    }


def postprocess(
    metadata: dict,
    top: Topology,
    flows: dict,
    cfg: Config = None,
    collective: str = None,
    write_schedule: bool = False,
    gurobi_env=None,
) -> dict:
    if cfg is None:
        cfg = Config()
    collective = collective or metadata["collective"]
    solver_type = metadata["solver_type"]
    T = metadata["T"]
    R = metadata["lp_rval"]

    g = top.construct(T)
    if gurobi_env is None:
        gurobi_env = make_gurobi_env(cfg)

    output_dict = {
        "objective_value": metadata["objective_value"],
        "solve_time": metadata["solve_time"],
    }

    if solver_type == "tree":
        f, w = flows["f"], flows["w"]
        if collective == "reduce_scatter":
            # The RS solution is a fan-in flow; time-reverse it into a
            # broadcast-from-sink so it decomposes as an out-tree (branching
            # restricted to reduce_nodes via determine_branching).
            g, f, w = reverse_time_expanded(g, f, w, T)
        ts, analysis = _tree_schedule(
            collective, top, g, f, w, R, gurobi_env, cfg, write_schedule=write_schedule
        )
        if ts is not None:
            output_dict["schedule"] = _interpret_tree_schedule(ts, collective)

    elif solver_type == "tree_paired":
        if collective != "all_reduce":
            logger.warning(
                "solver_type is tree_paired (all_reduce LP); ignoring collective override %r",
                collective,
            )
        f1, w1 = flows["f1"], flows["w1"]
        f2, w2 = flows["f2"], flows["w2"]
        tts, analysis = _allreduce_paired_schedule(
            top, g, f1, w1, f2, w2, R, gurobi_env, cfg, write_schedule=write_schedule
        )
        if tts is not None:
            output_dict["schedule"] = interpret_tts_as_all_reduce(tts)

    elif solver_type == "path":
        f = flows["f"]
        ps, analysis = _path_schedule(top, g, f, R, cfg, write_schedule=write_schedule)
        if ps is not None:
            output_dict["schedule"] = interpret_ps_as_all_to_all(ps)

    else:
        return output_dict

    if "schedule" in output_dict:
        output_dict["schedule_expanded"] = cfg.expand_schedule_symmetry
        if not cfg.expand_schedule_symmetry:
            output_dict["ec_expansion"] = ec_expansion_metadata(top)

    phase_factor = 2 if (solver_type == "tree" and collective == "all_reduce") else 1
    output_dict["phase_factor"] = phase_factor

    # output_dict["theoretical_step_length"] = analysis["step_len"]
    output_dict["n_steps"] = analysis["n_steps"] * phase_factor
    output_dict["ideal_latency"] = analysis["ideal_latency"] * phase_factor
    output_dict["realized_latency"] = (
        analysis["realized_latency_variable"] * phase_factor
    )
    output_dict["overhead"] = analysis["overhead_variable"]
    # output_dict["realized_latency_uniform_step"] = analysis["realized_latency"] * phase_factor
    # output_dict["overhead_uniform_step"] = analysis["overhead"]
    # output_dict["step_load_analysis"] = analysis

    realized_var = analysis["realized_latency_variable"] * phase_factor
    num_gpus = top.n
    data_size_kB = cfg.total_data_size
    output_dict["measured_latency_s"] = realized_var / 1e6
    output_dict["algorithmic_bandwidth_GBps"] = (
        num_gpus * (data_size_kB / 1e6) / output_dict["measured_latency_s"]
        if realized_var > 0
        else 0.0
    )

    # Per-step start times (cumulative variable step lengths, in seconds).
    step_lengths = analysis.get("step_lengths", {})
    sched = output_dict.get("schedule")
    if sched is not None and step_lengths:
        ns = max(step_lengths)
        if collective == "reduce_scatter":
            orig_of = lambda s: ns - s
        elif phase_factor == 2:
            orig_of = lambda s: ns - s if s <= ns else s - ns - 1
        else:
            orig_of = lambda s: s
        start = 0.0
        timed = {}
        for s in sorted(sched.keys()):
            timed[s] = {"start_time": start, "transmissions": sched[s]}
            start += step_lengths.get(orig_of(s), 0.0) / 1e6
        output_dict["schedule"] = timed

    logger.info(
        "Step-schedule overhead analysis:\n%s", format_step_load_report(analysis)
    )

    return output_dict
