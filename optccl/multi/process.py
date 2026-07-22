from collections import defaultdict

import numpy as np

from ..tree_decomposer import decompose_trees, reverse_time_expanded, _build_bc_coef
from ..process_results import (
    post_process_trees,
    decompose_and_process_paths,
    tree_bc_step_load,
    analyze_bc_step_load,
)

#: Joint-mode decomposition order of the demand families: p2p paths are an
#: exact flow decomposition (no repacking budget), then gather and reduce trees
#: each pack against the residual bandwidth of everything before them.
FAMILY_ORDER = ("p2p", "g", "r")


# --- projections: tagged flow keys -> the shapes the decomposers expect --------


def project_p2p(f_sol):
    f_p2p = {}
    for key, val in f_sol.items():
        if key[-1] == "p2p":
            i, j, t, o, d, _tag = key
            f_p2p[i, j, t, o, d] = val
    return f_p2p


def project_gather(f_sol, w_sol):
    f_g = defaultdict(float)
    for key, val in f_sol.items():
        # gather f key: (i, j, t, o, d, "g", idx)
        if len(key) == 7 and key[-2] == "g":
            i, j, t, o, d, _tag, _idx = key
            f_g[i, j, t, o, d] += val
    return dict(f_g), project_w(w_sol, "g")


def project_reduce(f_sol, w_sol):
    f_r = defaultdict(float)
    for key, val in f_sol.items():
        if len(key) == 7 and key[-2] == "r":
            i, j, t, o, d, _tag, _idx = key
            f_r[i, j, t, d, o] += val
    return dict(f_r), project_w(w_sol, "r")


def project_w(w_sol, tag):
    out = defaultdict(float)
    if tag == "p2p":
        for key, val in w_sol.items():
            if key[-1] == "p2p":
                i, j, t, o, d, _tag = key
                out[i, j, t, o] += val
    else:
        for key, val in w_sol.items():
            if len(key) == 6 and key[-2] == tag:
                i, j, t, o, _tag, _idx = key
                out[i, j, t, o] += val
    return dict(out)


# --- residual bandwidth ---------------------------------------------------------


def residual_rhs(top, g, w_sol, R, prior_tags):
    eidx = {e: i for i, e in enumerate(g.edges())}
    edge_list = list(eidx)
    bc_list = top.bandwidth_constraints
    bc_bounds = np.array([bc.bound for bc in bc_list], dtype=float)
    bc_edge_sets = [set(bc.edges) for bc in bc_list]

    w_tags = {tag: project_w(w_sol, tag) for tag in prior_tags}
    load = np.zeros(len(bc_list))
    for ec in top.ECs:
        o = ec.gpus[0]
        bc_coef = _build_bc_coef(ec, bc_edge_sets, edge_list, len(bc_list))
        w_caps = np.zeros(len(edge_list))
        for ei, ej in g.edges():
            i, t = ei
            j, _ = ej
            w_caps[eidx[(ei, ej)]] = sum(
                wt.get((i, j, t, o), 0.0) for wt in w_tags.values()
            )
        load += bc_coef.T @ w_caps
    return bc_bounds * R - load


# --- per-family processing -------------------------------------------------------


def process_p2p(top, f_sol, demand, C, K, R, cfg, T):
    g = top.construct(T)
    f_p2p = project_p2p(f_sol)
    paths = decompose_and_process_paths(top, g, f_p2p, cfg)
    return tree_bc_step_load(paths, top, demand, C, K, R)


def _tree_bc_load(
    top,
    g_use,
    f_use,
    w_use,
    demand,
    C,
    K,
    R,
    gurobi_env,
    cfg,
    bc_rhs_override,
    full_budget=True,
    collective="all_gather",
):
    active = {key[3] for key in f_use}
    all_ecs = top.ECs
    top.ECs = [ec for ec in all_ecs if ec.gpus[0] in active]
    try:
        decomposed = decompose_trees(
            top,
            g_use,
            f_use,
            w_use,
            collective,
            gurobi_env,
            cfg,
            R,
            full_budget=full_budget,
            bc_rhs_override=bc_rhs_override,
        )
        trees = post_process_trees(decomposed, set(top.components) - top.storage_nodes)
        trees = {
            o: [(el, vol * decomposed[o][1], d) for (el, vol, d) in tlist]
            for o, tlist in trees.items()
        }
        return tree_bc_step_load(trees, top, demand, C, K, R)
    finally:
        top.ECs = all_ecs


def process_gather(
    top,
    f_sol,
    w_sol,
    demand,
    C,
    K,
    R,
    gurobi_env,
    cfg,
    T,
    bc_rhs_override=None,
    full_budget=True,
):
    g = top.construct(T)
    f_g, w_g = project_gather(f_sol, w_sol)
    return _tree_bc_load(
        top,
        g,
        f_g,
        w_g,
        demand,
        C,
        K,
        R,
        gurobi_env,
        cfg,
        bc_rhs_override,
        full_budget=full_budget,
    )


def process_reduce(
    top,
    f_sol,
    w_sol,
    demand,
    C,
    K,
    R,
    gurobi_env,
    cfg,
    T,
    bc_rhs_override=None,
    full_budget=True,
):
    g = top.construct(T)
    f_r, w_r = project_reduce(f_sol, w_sol)
    g_rev, f_rev, w_rev = reverse_time_expanded(g, f_r, w_r, T)
    return _tree_bc_load(
        top,
        g_rev,
        f_rev,
        w_rev,
        demand,
        C,
        K,
        R,
        gurobi_env,
        cfg,
        bc_rhs_override,
        full_budget=full_budget,
        collective="reduce_scatter",
    )


PROCESS_FAMILY = {
    "g": process_gather,
    "r": process_reduce,
}


# --- combining + reporting --------------------------------------------------------


def merge_step_aligned(tables):
    merged = defaultdict(lambda: defaultdict(float))
    for tbl in tables:
        for bc, step_load in tbl.items():
            for s, load in step_load.items():
                merged[bc][s] += load
    return merged


def _scalars(analysis):
    return {
        "realized_latency": analysis["realized_latency_variable"],
        "ideal_latency": analysis["ideal_latency"],
        "overhead": analysis["overhead_variable"],
        "n_steps": analysis["n_steps"],
        "measured_latency_s": analysis["realized_latency_variable"] / 1e6,
    }


def summarize(bc_step_load, n_steps):
    analysis = analyze_bc_step_load(bc_step_load, n_steps)
    return _scalars(analysis), analysis
