import numpy as np

from collections import defaultdict

from .topologies import Topology
from .config import Config
from .dw import mirrored_dw


def build_reduce_family(top: Topology, T, subproblems, g, edgelist, f, w):
    for edge in g.edges():
        v1, v2 = edge
        i, t = v1
        j, _ = v2

        for ec in top.ECs:
            d = ec.gpus[0]

            wvar = subproblems[d].addVar(name=f"w2({i, j, t, d})")
            w[i, j, t, d] = wvar

            for o in top.gpus:
                fvar = subproblems[d].addVar(name=f"f2({i, j, t, o, d})")
                f[i, j, t, o, d] = fvar
                subproblems[d].addConstr(
                    w[i, j, t, d] >= f[i, j, t, o, d], f"xf2 {i, j, t, o, d}"
                )

    flow_dict = defaultdict(int)  # (node, time, origin, destination)
    for o in top.gpus:
        for d in top.gpus:
            flow_dict[o, 0, o, d] = 1

    for o in top.gpus:
        for d in top.gpus:
            flow_dict[d, T, o, d] = -1

    for ec in top.ECs:
        d = ec.gpus[0]
        for o in top.gpus:
            for node in g.nodes():
                i, t = node
                subproblems[d].addConstr(
                    sum(f[e1[0], e2[0], e1[1], o, d] for e1, e2 in g.out_edges(node))
                    - sum(f[e1[0], e2[0], e1[1], o, d] for e1, e2 in g.in_edges(node))
                    == flow_dict[i, t, o, d],
                    f"flow2 {i, t, o, d}",
                )

    for ec in top.ECs:
        d = ec.gpus[0]
        for node in g.nodes():
            i, t = node

            if i not in top.reduce_nodes:
                subproblems[d].addConstr(
                    sum(w[e1[0], e2[0], e1[1], d] for e1, e2 in g.out_edges(node))
                    == sum(w[e1[0], e2[0], e1[1], d] for e1, e2 in g.in_edges(node)),
                    f"w2 pass through {i, t, d}",
                )
            else:
                out_total = sum(
                    w[e1[0], e2[0], e1[1], d] for e1, e2 in g.out_edges(node)
                )
                for ie1, ie2 in g.in_edges(node):
                    subproblems[d].addConstr(
                        out_total - min(flow_dict[i, t, o, d] for o in top.gpus)
                        >= w[ie1[0], ie2[0], ie1[1], d],
                        f"w2 stability {i, t, d, ie2[1]}",
                    )


class _ReduceScatterFormulation:
    def __init__(self, top: Topology, T):
        self.top = top
        self.T = T
        self.f = {}  # flow variables (tail, head, starttime, origin, destination)
        self.w = {}  # usage variables. Used to minimize number of linking variables
        self.edgelist = None

    def build(self, subproblems, g, edgelist):
        self.edgelist = edgelist
        build_reduce_family(self.top, self.T, subproblems, g, edgelist, self.f, self.w)

    def priced_groups(self, o):
        return [[self.w[ei, ej, t, o] for ((ei, t), (ej, _)) in self.edgelist]]

    def f_values(self, o):
        # Edge-major x source, keyed with the representative as sink.
        return np.array(
            [
                self.f[ei, ej, t, d, o].X
                for ((ei, t), (ej, _)) in self.edgelist
                for d in self.top.gpus
            ]
        )


def reduce_scatter_mirrored_dw(top: Topology, T, R, gurobi_env, cfg: Config):
    formulation = _ReduceScatterFormulation(top, T)
    m, w_arrays, f_arrays, g, elapsed, R = mirrored_dw(
        top,
        T,
        R,
        gurobi_env,
        formulation,
        name="reduce_scatter",
        no_crossover=cfg.rs_no_crossover,
        phase1_alpha=cfg.rs_phase1_cost_weight,
        feasible_only=cfg.stop_at_feasible,
        dw_tolerance=cfg.dw_tolerance,
        rate_retry_max=cfg.rate_retry_max,
        rate_retry_factor=cfg.rate_retry_factor,
    )
    edgelist = formulation.edgelist

    f_sol = {}
    w_sol = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        if o not in w_arrays:
            continue
        array_sol = f_arrays[o]
        array_w_sol = w_arrays[o]

        counter = 0
        for (ei, t), (ej, _) in edgelist:
            for d in top.gpus:
                f_sol[ei, ej, t, d, o] = array_sol[counter]
                counter += 1

        counter = 0
        for (ei, t), (ej, _) in edgelist:
            w_sol[ei, ej, t, o] = array_w_sol[counter]
            counter += 1
    return m, f_sol, w_sol, g, elapsed, R
