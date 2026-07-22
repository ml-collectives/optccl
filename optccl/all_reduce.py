import numpy as np

from collections import defaultdict

from .topologies import Topology
from .config import Config
from .dw import mirrored_dw
from .reduce_scatter import build_reduce_family


class _AllReduceFormulation:
    def __init__(self, top: Topology, T):
        self.top = top
        self.T = T
        self.f1 = {}  # flow variables (tail, head, starttime, origin, destination)
        self.f2 = {}
        self.w1 = {}  # usage variables. Used to minimize number of linking variables
        self.w2 = {}
        self.edgelist = None

    def build(self, subproblems, g, edgelist):
        top, T = self.top, self.T
        f1, w1 = self.f1, self.w1
        self.edgelist = edgelist

        # Create variables
        for edge in g.edges():
            v1, v2 = edge
            i, t = v1
            j, _ = v2

            for ec in top.ECs:
                o = ec.gpus[0]

                wvar = subproblems[o].addVar(name=f"w1({i, j, t, o})")
                w1[i, j, t, o] = wvar

                for d in top.gpus:
                    fvar = subproblems[o].addVar(name=f"f1({i, j, t, o, d})")
                    f1[i, j, t, o, d] = fvar
                    subproblems[o].addConstr(
                        w1[i, j, t, o] >= f1[i, j, t, o, d], f"xf1 {i, j, t, o, d}"
                    )

        flow_dict = defaultdict(int)  # (node, time, origin, destination)
        for o in top.gpus:
            for d in top.gpus:
                flow_dict[o, 0, o, d] = 1

        for o in top.gpus:
            for d in top.gpus:
                flow_dict[d, T, o, d] = -1

        for ec in top.ECs:
            o = ec.gpus[0]
            for d in top.gpus:
                for node in g.nodes():
                    i, t = node
                    subproblems[o].addConstr(
                        sum(
                            f1[e1[0], e2[0], e1[1], o, d]
                            for e1, e2 in g.out_edges(node)
                        )
                        - sum(
                            f1[e1[0], e2[0], e1[1], o, d] for e1, e2 in g.in_edges(node)
                        )
                        == flow_dict[i, t, o, d],
                        f"flow1 {i, t, o, d}",
                    )

        for ec in top.ECs:
            o = ec.gpus[0]
            for node in g.nodes():
                i, t = node

                if i not in top.copy_nodes:
                    subproblems[o].addConstr(
                        sum(w1[e1[0], e2[0], e1[1], o] for e1, e2 in g.out_edges(node))
                        == sum(
                            w1[e1[0], e2[0], e1[1], o] for e1, e2 in g.in_edges(node)
                        ),
                        f"w1 pass through {i, t, o}",
                    )
                else:
                    in_total = sum(
                        w1[e1[0], e2[0], e1[1], o] for e1, e2 in g.in_edges(node)
                    )
                    for oe1, oe2 in g.out_edges(node):
                        subproblems[o].addConstr(
                            in_total + min(flow_dict[i, t, o, d] for d in top.gpus)
                            >= w1[oe1[0], oe2[0], oe1[1], o],
                            f"w1 stability {i, t, o, oe2[1]}",
                        )

        # The reduce-scatter half (f2/w2, stability gated on reduce_nodes) is
        # shared with the standalone ReduceScatter formulation.
        build_reduce_family(top, T, subproblems, g, edgelist, self.f2, self.w2)

    def priced_groups(self, o):
        return [
            [self.w1[ei, ej, t, o] for ((ei, t), (ej, _)) in self.edgelist],
            [self.w2[ei, ej, t, o] for ((ei, t), (ej, _)) in self.edgelist],
        ]

    def f_values(self, o):
        # f1 then f2, each edge-major x destination (f2 keyed with the
        # representative as sink: (edge, t, origin=d, sink=o)).
        xp1 = [
            self.f1[ei, ej, t, o, d].X
            for ((ei, t), (ej, _)) in self.edgelist
            for d in self.top.gpus
        ]
        xp2 = [
            self.f2[ei, ej, t, d, o].X
            for ((ei, t), (ej, _)) in self.edgelist
            for d in self.top.gpus
        ]
        return np.array(xp1 + xp2)


def all_reduce_mirrored_dw(top: Topology, T, R, gurobi_env, cfg: Config):
    formulation = _AllReduceFormulation(top, T)
    m, w_arrays, f_arrays, g, elapsed, R = mirrored_dw(
        top,
        T,
        R,
        gurobi_env,
        formulation,
        name="all_reduce",
        no_crossover=cfg.ar_no_crossover,
        phase1_alpha=cfg.ar_phase1_cost_weight,
        feasible_only=cfg.stop_at_feasible,
        dw_tolerance=cfg.dw_tolerance,
        rate_retry_max=cfg.rate_retry_max,
        rate_retry_factor=cfg.rate_retry_factor,
    )
    edgelist = formulation.edgelist
    gsize = len(edgelist)
    fsize = gsize * len(top.gpus)

    f1_sol = {}
    w1_sol = {}
    f2_sol = {}
    w2_sol = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        if o not in w_arrays:
            continue
        array_f = f_arrays[o]
        array_w = w_arrays[o]

        counter = 0
        for (ei, t), (ej, _) in edgelist:
            for d in top.gpus:
                f1_sol[ei, ej, t, o, d] = array_f[counter]
                f2_sol[ei, ej, t, d, o] = array_f[fsize + counter]
                counter += 1

        counter = 0
        for (ei, t), (ej, _) in edgelist:
            w1_sol[ei, ej, t, o] = array_w[counter]
            w2_sol[ei, ej, t, o] = array_w[gsize + counter]
            counter += 1
    return m, f1_sol, w1_sol, f2_sol, w2_sol, g, elapsed, R
