import logging

import networkx as nx
from collections import defaultdict

from .strengthen import original_component
from .wfq import (
    build_tree_dag,
    physical_edges as _get_physical_edges,
    wfq_schedule,
    wfq_load_counts,
    wfq_allreduce_load_counts,
)

logger = logging.getLogger(__name__)


def _accumulate_bc_step_loads(full_schedule, top):
    """Aggregate transmitted bytes per (bandwidth-constraint, step)."""
    edge_to_bcs = defaultdict(list)
    for bc in top.bandwidth_constraints:
        for edge in bc.edges:
            edge_to_bcs[edge].append(bc)

    bc_step_load = defaultdict(lambda: defaultdict(float))
    for step, edge_map in full_schedule.items():
        for spatial_edge, transmissions in edge_map.items():
            for tx in transmissions:
                chunk_size = tx[1]
                switches = tx[2]
                for pe in _get_physical_edges(spatial_edge, switches):
                    for bc in edge_to_bcs.get(pe, []):
                        bc_step_load[bc][step] += chunk_size
    return bc_step_load


def analyze_step_loads(full_schedule, top, n_steps=None):
    """Attribute step-schedule overhead to a concrete (constraint, step)."""
    bc_step_load = _accumulate_bc_step_loads(full_schedule, top)
    if n_steps is None:
        n_steps = (max(full_schedule.keys()) + 1) if full_schedule else 0
    return analyze_bc_step_load(bc_step_load, n_steps)


def analyze_bc_step_load(bc_step_load, n_steps):
    """Compute the overhead attribution from a prebuilt ``{bc: {step: load}}`` table."""
    # Per-step bottleneck ratio = max over constraints of load/bound in that step.
    # Summing it gives the latency if each step ran at its OWN length (variable-step
    # cost) rather than every step taking as long as the global peak (uniform-step).
    step_peak = defaultdict(float)
    for bc, step_load in bc_step_load.items():
        for step, load in step_load.items():
            r = load / bc.bound
            if r > step_peak[step]:
                step_peak[step] = r
    realized_latency_variable = sum(step_peak.values())

    per_bc = []
    step_len = 0.0
    bottleneck_step = None
    bottleneck_bc = None
    ideal_latency = 0.0
    ideal_bc = None

    for bc, step_load in bc_step_load.items():
        if not step_load:
            continue
        total = sum(step_load.values())
        peak_step, peak = max(step_load.items(), key=lambda kv: kv[1])
        active = len(step_load)
        mean_active = total / active if active else 0.0
        bound = bc.bound
        peak_ratio = peak / bound
        mean_ratio = mean_active / bound
        burstiness = peak / mean_active if mean_active else 0.0
        bc_ideal = total / bound  # latency lower bound contributed by this constraint

        per_bc.append(
            {
                "name": bc.name,
                "bound": bound,
                "total_load": total,
                "peak_load": peak,
                "peak_step": peak_step,
                "active_steps": active,
                "mean_load": mean_active,
                "peak_ratio": peak_ratio,
                "mean_ratio": mean_ratio,
                "burstiness": burstiness,
                "ideal_latency": bc_ideal,
            }
        )

        if peak_ratio > step_len:
            step_len = peak_ratio
            bottleneck_step = peak_step
            bottleneck_bc = bc
        if bc_ideal > ideal_latency:
            ideal_latency = bc_ideal
            ideal_bc = bc

    per_bc.sort(key=lambda d: d["peak_ratio"], reverse=True)

    realized_latency = step_len * n_steps
    overhead = (realized_latency / ideal_latency - 1.0) if ideal_latency else 0.0
    overhead_variable = (
        (realized_latency_variable / ideal_latency - 1.0) if ideal_latency else 0.0
    )

    return {
        "step_len": step_len,
        "n_steps": n_steps,
        "realized_latency": realized_latency,
        "realized_latency_variable": realized_latency_variable,
        "ideal_latency": ideal_latency,
        "ideal_bc": ideal_bc.name if ideal_bc else None,
        "overhead": overhead,
        "overhead_variable": overhead_variable,
        "bottleneck_step": bottleneck_step,
        "bottleneck_bc": bottleneck_bc.name if bottleneck_bc else None,
        "bottleneck_burstiness": (
            next(
                (
                    d["burstiness"]
                    for d in per_bc
                    if d["name"] == (bottleneck_bc.name if bottleneck_bc else None)
                ),
                None,
            )
        ),
        "per_bc": per_bc,
        # Per-step variable length (max over constraints of load/bound, raw units); the
        # cumulative sum gives each step's start time. dict {step: length}.
        "step_lengths": dict(step_peak),
    }


def stream_tree_overhead(
    trees_by_origin, top, demand, C, K, R, *, required_chunks=None
):
    """Compute the step-schedule overhead WITHOUT materializing the full schedule."""
    edge_to_bcs = defaultdict(list)
    for bc in top.bandwidth_constraints:
        for e in bc.edges:
            edge_to_bcs[e].append(bc)

    bc_step_load = defaultdict(lambda: defaultdict(float))
    max_step = [-1]

    for ec in top.ECs:
        o = ec.gpus[0]
        n = len(ec.gpus)

        edge_step_load = defaultdict(lambda: defaultdict(float))

        def emit(t, se, tid, chunk_id, pes, esl=edge_step_load):
            if t > max_step[0]:
                max_step[0] = t
            for pe in pes:
                esl[pe][t] += C

        wfq_schedule(
            trees_by_origin[o],
            demand,
            C,
            K,
            R,
            required_chunks=required_chunks,
            emit=emit,
        )

        for pe, tload in edge_step_load.items():
            targets = []
            for i in range(n):
                pe_sh = (ec.shift_fn(pe[0], i), ec.shift_fn(pe[1], i))
                targets.extend(edge_to_bcs.get(pe_sh, []))
            for bc in targets:
                fl = bc_step_load[bc]
                for t, load in tload.items():
                    fl[t] += load

    n_steps = max_step[0] + 1 if max_step[0] >= 0 else 0
    return analyze_bc_step_load(bc_step_load, n_steps)


def _expand_edge_load_over_ec(edge_step_load, ec, edge_to_bcs, bc_step_load):
    """Add one origin's per-(physical edge, step) load to bc_step_load, expanded over the
    EC shift group at the edge level (shift commutes with physical-edge expansion)."""
    n = len(ec.gpus)
    for pe, tload in edge_step_load.items():
        targets = []
        for i in range(n):
            pe_sh = (ec.shift_fn(pe[0], i), ec.shift_fn(pe[1], i))
            targets.extend(edge_to_bcs.get(pe_sh, []))
        for bc in targets:
            fl = bc_step_load[bc]
            for t, load in tload.items():
                fl[t] += load


def tree_bc_step_load(trees_by_origin, top, demand, C, K, R):
    """Per-(bandwidth-constraint, step) load table for a tree (or path) decomposition."""
    edge_to_bcs = defaultdict(list)
    for bc in top.bandwidth_constraints:
        for e in bc.edges:
            edge_to_bcs[e].append(bc)

    bc_step_load = defaultdict(lambda: defaultdict(float))
    n_steps = 0
    for ec in top.ECs:
        o = ec.gpus[0]
        edge_load, ns = wfq_load_counts(trees_by_origin[o], demand, C, K, R)
        n_steps = max(n_steps, ns)
        _expand_edge_load_over_ec(edge_load, ec, edge_to_bcs, bc_step_load)

    return bc_step_load, n_steps


def count_tree_overhead(trees_by_origin, top, demand, C, K, R):
    """Overhead via the count-based WFQ sim."""
    bc_step_load, n_steps = tree_bc_step_load(trees_by_origin, top, demand, C, K, R)
    return analyze_bc_step_load(bc_step_load, n_steps)


def count_allreduce_overhead(
    rs_trees_by_origin, ag_trees_by_origin, top, demand, C, K, R
):
    """Count-based overhead for the all_reduce (paired reduce-scatter + all-gather) pipeline."""
    edge_to_bcs = defaultdict(list)
    for bc in top.bandwidth_constraints:
        for e in bc.edges:
            edge_to_bcs[e].append(bc)

    bc_step_load = defaultdict(lambda: defaultdict(float))
    n_steps = 0
    for ec in top.ECs:
        o = ec.gpus[0]
        edge_load, ns = wfq_allreduce_load_counts(
            rs_trees_by_origin[o], ag_trees_by_origin[o], demand, C, K, R
        )
        n_steps = max(n_steps, ns)
        _expand_edge_load_over_ec(edge_load, ec, edge_to_bcs, bc_step_load)

    return analyze_bc_step_load(bc_step_load, n_steps)


def count_path_overhead(paths_by_origin, top, demand, C, K, R):
    """Count-based overhead for the all_to_all (path) pipeline."""
    return count_tree_overhead(paths_by_origin, top, demand, C, K, R)


def format_step_load_report(analysis, top_n=8):
    """Render an analyze_step_loads() result as a human-readable diagnostic block."""
    lines = []
    lines.append(
        f"  step_len (peak load/bound) : {analysis['step_len']:.4f}  "
        f"@ step {analysis['bottleneck_step']} on '{analysis['bottleneck_bc']}'"
    )
    if analysis["bottleneck_burstiness"] is not None:
        lines.append(
            f"  bottleneck burstiness      : {analysis['bottleneck_burstiness']:.3f}x (peak/mean)"
        )
    lines.append(f"  n_steps                    : {analysis['n_steps']}")
    lines.append(f"  realized latency           : {analysis['realized_latency']:.4f}")
    lines.append(
        f"  ideal latency (lower bound): {analysis['ideal_latency']:.4f}  "
        f"(set by '{analysis['ideal_bc']}')"
    )
    lines.append(
        f"  >>> overhead (variable step): {analysis['overhead_variable'] * 100:.1f}%  "
        f"(sum of per-step peaks; each step costs its own length)"
    )
    lines.append(
        f"      overhead (uniform step) : {analysis['overhead'] * 100:.1f}%  "
        f"(step_len * n_steps; every step costs the global peak)"
    )
    lines.append(f"  top {top_n} constraints by peak/bound:")
    lines.append(
        f"    {'name':<18} {'peak/bnd':>9} {'mean/bnd':>9} {'burst':>7} {'pk_step':>8} {'active':>7}"
    )
    for d in analysis["per_bc"][:top_n]:
        lines.append(
            f"    {d['name'][:18]:<18} {d['peak_ratio']:>9.3f} {d['mean_ratio']:>9.3f} "
            f"{d['burstiness']:>7.2f} {str(d['peak_step']):>8} {d['active_steps']:>7}"
        )
    return "\n".join(lines)


def compute_theoretical_step_length(full_schedule, top):
    bc_step_load = _accumulate_bc_step_loads(full_schedule, top)

    max_ratio = 0.0
    bottleneck_step = None
    bottleneck_bc = None
    for bc, step_load in bc_step_load.items():
        for step, load in step_load.items():
            ratio = load / bc.bound
            if ratio > max_ratio:
                max_ratio = ratio
                bottleneck_step = step
                bottleneck_bc = bc

    return max_ratio, bottleneck_step, bottleneck_bc


def post_process_trees(decomposed_trees, switches):
    logger.info("Post processing")
    wfq_input_trees = {}  # origin -> tree list
    switches = set(switches)
    for o, dtr in decomposed_trees.items():
        sol, obj = dtr

        wfq_input_trees[o] = []

        for tree in sol:
            edge_dict, volume, suffixes, edge_copy_dests = tree
            edge_list = []
            edge_dests = []

            for d, edges in suffixes.items():
                edge_dests.append(d[0])

                # Keep track of destinations at head of edge
                dest_map = {}
                for u, v in edges:
                    for dest_set in edge_copy_dests[(u, v)]:
                        if d[0] in dest_set:
                            dest_map[v] = dest_set
                            break
                    assert v in dest_map, (
                        f"{edge_copy_dests}, \n\n {edges}, \n\n {v}, \n\n {edge_dict}"
                    )

                adj = {v: u for u, v in edges}  # track backwards

                curr = d
                back_path = [curr]
                while curr in adj:
                    next_node = adj[curr]
                    back_path.append(next_node)
                    curr = next_node

                real_path = [back_path[-1]]
                forward_path = list(reversed(back_path))

                switch_list = []
                switch_dict = {}

                for i in range(1, len(forward_path)):
                    node = forward_path[i]
                    if node[0] in switches and i < len(forward_path) - 1:
                        switch_list.append(node[0])
                    else:
                        switch_dict[real_path[-1], node] = switch_list
                        switch_list = []
                        real_path.append(node)

                # Through one last time to construct tree tuples. Strengthening's
                # per-layer (storage) duplicates collapse back to their original
                # component here: they ARE that vertex, just split in the LP
                for i in range(1, len(real_path)):
                    (u, t) = real_path[i - 1]
                    (v, t2) = real_path[i]
                    edge_tuple = (
                        original_component(u),
                        original_component(v),
                        t,
                        {
                            "switches": switch_dict[(u, t), (v, t2)],
                            "dests": dest_map[(v, t2)],
                        },
                    )
                    edge_list.append(edge_tuple)

            new_volume = volume * 1 / obj

            tree_tuple = (edge_list, new_volume, edge_dests)
            wfq_input_trees[o].append(tree_tuple)

    return wfq_input_trees


def decompose_and_process_paths(top, g, f, cfg):
    logger.info("Decomposing and processing paths")
    tolerance = cfg.decomposition_tolerance

    T = -1
    edgelist = list(g.edges())
    for v1, v2 in edgelist:
        _, t2 = v2
        T = max(T, t2)

    # Build residual graphs for each (o, d) pair present in f
    od_pairs = set()
    for key in f.keys():
        i, j, t, o, d = key
        if o != d:
            od_pairs.add((o, d))

    residual_graphs = {}
    for od in od_pairs:
        residual_graphs[od] = nx.DiGraph()
        residual_graphs[od].add_nodes_from(g.nodes())

    for v1, v2 in edgelist:
        i, t = v1
        j, _ = v2
        for o, d in od_pairs:
            fval = f.get((i, j, t, o, d), 0)
            if fval > tolerance:
                residual_graphs[(o, d)].add_edge(v1, v2, flow=fval)

    processed_paths = {}

    for ec in top.ECs:
        o = ec.gpus[0]
        ec_paths = []

        for d in top.gpus:
            if d == o:
                continue

            source = (o, 0)
            sink = (d, T)
            residual = residual_graphs.get((o, d))
            if residual is None:
                continue

            while True:
                try:
                    path = nx.shortest_path(residual, source=source, target=sink)
                except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
                    break

                if len(path) < 2:
                    break

                path_flow = min(
                    residual[u][v]["flow"] for u, v in zip(path[:-1], path[1:])
                )

                if path_flow < tolerance:
                    break

                # Remove switch nodes (same time step): build real_path and record switches
                real_path = [path[0]]
                switch_list = []
                switch_dict = {}

                for idx in range(1, len(path)):
                    node = path[idx]
                    prev_real = real_path[-1]
                    if node[1] == prev_real[1] and idx < len(path) - 1:
                        switch_list.append(node[0])
                    else:
                        switch_dict[prev_real, node] = switch_list
                        switch_list = []
                        real_path.append(node)

                edge_list = []
                for idx in range(1, len(real_path)):
                    (u, t_u) = real_path[idx - 1]
                    (v, t_v) = real_path[idx]
                    edge_list.append(
                        (u, v, t_u, {"switches": switch_dict[(u, t_u), (v, t_v)]})
                    )

                ec_paths.append((edge_list, path_flow, d))

                # Subtract path flow from residual
                edges_to_remove = []
                for u, v in zip(path[:-1], path[1:]):
                    residual[u][v]["flow"] -= path_flow
                    if residual[u][v]["flow"] < tolerance:
                        edges_to_remove.append((u, v))
                residual.remove_edges_from(edges_to_remove)

        processed_paths[o] = ec_paths

    return processed_paths


def _n_shifts(ec, expand):
    return len(ec.gpus) if expand else 1


def ec_expansion_metadata(top):
    """Per EC, everything needed to expand a representative-only schedule downstream:
    the representative origin, its member GPUs, and the component permutation induced by
    each shift (image lists parallel to ``nodes``)."""
    nodes = list(top.components)
    return [
        {
            "representative": ec.gpus[0],
            "gpus": list(ec.gpus),
            "nodes": nodes,
            "shifts": [[ec.shift_fn(c, i) for c in nodes] for i in range(len(ec.gpus))],
        }
        for ec in top.ECs
    ]


def collect_wfq_scheds(scheds, chunk_size, ECs, expand=True):
    logger.info("Collecting schedules")
    full_schedule = defaultdict(lambda: defaultdict(list))

    for ec in ECs:
        o = ec.gpus[0]
        chunk_mapping = {}

        total = sum(
            len(scheds[o][step][edge]) for step in scheds[o] for edge in scheds[o][step]
        )
        logger.debug(
            "Source transmissions: %d, GPUs: %d, ECs: %d", total, len(ec.gpus), len(ECs)
        )

        for i in range(_n_shifts(ec, expand)):
            for step in scheds[o]:
                for edge in scheds[o][step]:
                    ei, ej = edge
                    mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))

                    for transmission in scheds[o][step][edge]:
                        tree_id = transmission["tree_id"]
                        chunk_id = transmission["chunk_id"]
                        switches = transmission["metadata"]["switches"]
                        dests = transmission["metadata"]["dests"]

                        if (tree_id, chunk_id) in chunk_mapping:
                            chunk_no = chunk_mapping[tree_id, chunk_id]
                        else:
                            chunk_no = len(chunk_mapping)
                            chunk_mapping[tree_id, chunk_id] = chunk_no

                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dests = {ec.shift_fn(dest, i) for dest in dests}
                        full_schedule[step][mapped_edge].append(
                            (chunk_no, chunk_size, mapped_switches, mapped_dests)
                        )

    return full_schedule


def collect_raw_tree_scheds(trees_by_origin, demand, K, ECs, expand=True):
    # first, do the pipelining.
    sched = {}
    for o, trees in trees_by_origin.items():
        o_schedule = defaultdict(lambda: defaultdict(list))
        chunk_no = 0
        for tree in trees:
            edge_list, volume, edge_dests = tree

            _, _, _, _, lanes, _ = build_tree_dag(edge_list)

            chunk_size = volume * demand / K
            for i in range(K):
                for lane_id, lane in lanes.items():
                    edg = lane["spatial"]
                    t = lane["time"]
                    md = lane["metadata"]

                    o_schedule[i + t][edg].append(
                        (chunk_no, chunk_size, md["switches"], md["dests"])
                    )
                chunk_no += 1
        sched[o] = o_schedule

    full_schedule = defaultdict(lambda: defaultdict(list))

    for ec in ECs:
        o = ec.gpus[0]
        for i in range(_n_shifts(ec, expand)):
            for step in sched[o]:
                for edge in sched[o][step]:
                    ei, ej = edge
                    mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))

                    for transmission in sched[o][step][edge]:
                        chunk_no, chunk_size, switches, dests = transmission
                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dests = {ec.shift_fn(dest, i) for dest in dests}
                        full_schedule[step][mapped_edge].append(
                            (chunk_no, chunk_size, mapped_switches, mapped_dests)
                        )

    return full_schedule


def collect_wfq_path_scheds(scheds, chunk_size, ECs, expand=True):
    logger.info("Collecting path schedules")
    full_schedule = defaultdict(lambda: defaultdict(list))

    for ec in ECs:
        o = ec.gpus[0]
        chunk_mapping = {}

        total = sum(
            len(scheds[o][step][edge]) for step in scheds[o] for edge in scheds[o][step]
        )
        logger.debug(
            "Source transmissions: %d, GPUs: %d, ECs: %d", total, len(ec.gpus), len(ECs)
        )

        for i in range(_n_shifts(ec, expand)):
            for step in scheds[o]:
                for edge in scheds[o][step]:
                    ei, ej = edge
                    mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))

                    for transmission in scheds[o][step][edge]:
                        path_id = transmission["tree_id"]
                        chunk_id = transmission["chunk_id"]
                        switches = transmission["metadata"]["switches"]
                        dest = transmission["metadata"]["dest"]

                        if (path_id, chunk_id) in chunk_mapping:
                            chunk_no = chunk_mapping[path_id, chunk_id]
                        else:
                            chunk_no = len(chunk_mapping)
                            chunk_mapping[path_id, chunk_id] = chunk_no

                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dest = {ec.shift_fn(dest, i)}
                        full_schedule[step][mapped_edge].append(
                            (chunk_no, chunk_size, mapped_switches, mapped_dest)
                        )

    return full_schedule


def collect_raw_path_scheds(paths_by_origin, demand, K, ECs, expand=True):
    sched = {}
    for o, paths in paths_by_origin.items():
        o_schedule = defaultdict(lambda: defaultdict(list))
        chunk_no = 0

        for edge_list, volume, dest in paths:
            chunk_size = volume * demand / K
            for i in range(K):
                for u, v, t, meta in edge_list:
                    o_schedule[i + t][(u, v)].append(
                        (chunk_no, chunk_size, meta["switches"], {dest})
                    )
            chunk_no += 1

        sched[o] = o_schedule

    full_schedule = defaultdict(lambda: defaultdict(list))

    for ec in ECs:
        o = ec.gpus[0]
        for i in range(_n_shifts(ec, expand)):
            for step in sched[o]:
                for edge in sched[o][step]:
                    ei, ej = edge
                    mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))

                    for transmission in sched[o][step][edge]:
                        chunk_no, chunk_size, switches, dests = transmission
                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dests = {ec.shift_fn(d, i) for d in dests}
                        full_schedule[step][mapped_edge].append(
                            (chunk_no, chunk_size, mapped_switches, mapped_dests)
                        )

    return full_schedule


def collect_allreduce_wfq_scheds(rs_scheds, ag_scheds, chunk_size, ECs, expand=True):
    """Combine the reduce-scatter and all-gather phases into one executable schedule."""
    logger.info("Collecting allreduce schedules")
    full_schedule = defaultdict(lambda: defaultdict(list))

    max_rs = max(
        (step for o_sched in rs_scheds.values() for step in o_sched), default=-1
    )
    ag_offset = max_rs + 1  # all-gather begins after reduce-scatter completes

    for ec in ECs:
        o = ec.gpus[0]
        for i in range(_n_shifts(ec, expand)):
            # Reduce-scatter: reversed (time within [0, max_rs], edge direction, switch order).
            for step, edge_map in rs_scheds.get(o, {}).items():
                out_step = max_rs - step
                for (ei, ej), transmissions in edge_map.items():
                    mapped_edge = (ec.shift_fn(ej, i), ec.shift_fn(ei, i))
                    for tx in transmissions:
                        switches = tx["metadata"]["switches"][::-1]
                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dests = {
                            ec.shift_fn(d, i) for d in tx["metadata"]["dests"]
                        }
                        full_schedule[out_step][mapped_edge].append(
                            (
                                tx["global_id"],
                                chunk_size,
                                mapped_switches,
                                mapped_dests,
                                "rs",
                            )
                        )
            # All-gather: forward, offset to run after the reduce-scatter.
            for step, edge_map in ag_scheds.get(o, {}).items():
                out_step = step + ag_offset
                for (ei, ej), transmissions in edge_map.items():
                    mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))
                    for tx in transmissions:
                        switches = tx["metadata"]["switches"]
                        mapped_switches = [ec.shift_fn(sw, i) for sw in switches]
                        mapped_dests = {
                            ec.shift_fn(d, i) for d in tx["metadata"]["dests"]
                        }
                        full_schedule[out_step][mapped_edge].append(
                            (
                                tx["global_id"],
                                chunk_size,
                                mapped_switches,
                                mapped_dests,
                                "ag",
                            )
                        )

    return full_schedule


def collect_raw_allreduce_scheds(
    rs_trees_by_origin, ag_trees_by_origin, demand, K, ECs, expand=True
):
    rs_sched = {}
    ag_sched = {}

    for o, rs_trees in rs_trees_by_origin.items():
        ag_trees = ag_trees_by_origin.get(o, [])

        rs_o = defaultdict(lambda: defaultdict(list))
        ag_o = defaultdict(lambda: defaultdict(list))

        chunk_no = 0
        T_rs = 0
        for edge_list, volume, _ in rs_trees:
            _, _, _, _, lanes, _ = build_tree_dag(edge_list)
            chunk_size = volume * demand / K
            for i in range(K):
                for lane in lanes.values():
                    step = i + lane["time"]
                    T_rs = max(T_rs, step + 1)
                    rs_o[step][lane["spatial"]].append(
                        (
                            chunk_no,
                            chunk_size,
                            lane["metadata"]["switches"],
                            lane["metadata"]["dests"],
                            "rs",
                        )
                    )
            chunk_no += 1

        chunk_no_ag = 0
        for edge_list, volume, _ in ag_trees:
            _, _, _, _, lanes, _ = build_tree_dag(edge_list)
            chunk_size = volume * demand / K
            for i in range(K):
                for lane in lanes.values():
                    step = T_rs + i + lane["time"]
                    ag_o[step][lane["spatial"]].append(
                        (
                            chunk_no_ag,
                            chunk_size,
                            lane["metadata"]["switches"],
                            lane["metadata"]["dests"],
                            "ag",
                        )
                    )
            chunk_no_ag += 1

        rs_sched[o] = rs_o
        ag_sched[o] = ag_o

    full_schedule = defaultdict(lambda: defaultdict(list))
    for ec in ECs:
        o = ec.gpus[0]
        for i in range(_n_shifts(ec, expand)):
            for phase_tag, sched in (("rs", rs_sched), ("ag", ag_sched)):
                for step, edge_map in sched.get(o, {}).items():
                    for edge, transmissions in edge_map.items():
                        ei, ej = edge
                        mapped_edge = (ec.shift_fn(ei, i), ec.shift_fn(ej, i))
                        for (
                            chunk_no,
                            chunk_size,
                            switches,
                            dests,
                            phase,
                        ) in transmissions:
                            full_schedule[step][mapped_edge].append(
                                (
                                    chunk_no,
                                    chunk_size,
                                    [ec.shift_fn(sw, i) for sw in switches],
                                    {ec.shift_fn(d, i) for d in dests},
                                    phase,
                                )
                            )

    return full_schedule


def _stringify_schedule(sched):
    return {
        step: {str(edge): txns for edge, txns in edge_map.items()}
        for step, edge_map in sched.items()
    }


def interpret_ts_as_all_gather(ts):
    return _stringify_schedule(ts)


def interpret_ts_as_reduce_scatter(ts):
    full_schedule = defaultdict(lambda: defaultdict(list))

    num_steps = max(ts.keys())

    for step, edge_maps in ts.items():
        for edge, transmissions in edge_maps.items():
            u, v = edge
            for cn, cs, sws, ds in transmissions:
                rsw = sws[::-1]
                full_schedule[num_steps - step][str((v, u))].append((cn, cs, rsw, ds))
    return full_schedule


def interpret_ts_as_all_reduce(ts):
    full_schedule = defaultdict(lambda: defaultdict(list))

    num_steps = max(ts.keys())

    for step, edge_maps in ts.items():
        for edge, transmissions in edge_maps.items():
            u, v = edge
            for cn, cs, sws, ds in transmissions:
                rsws = sws[::-1]
                full_schedule[num_steps - step][str((v, u))].append((cn, cs, rsws, ds))
                full_schedule[num_steps + step + 1][str(edge)].append((cn, cs, sws, ds))
    return full_schedule


def interpret_ps_as_all_to_all(ps):
    return _stringify_schedule(ps)


def interpret_tts_as_all_reduce(tts):
    return _stringify_schedule(tts)
