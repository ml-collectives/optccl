import logging

import networkx as nx
import gurobipy as gp
import numpy as np

from collections import defaultdict

from .topologies import Topology
from .config import Config
from .dw import mirrored_dw
from .errors import OptcclError

logger = logging.getLogger(__name__)


class _AllGatherFormulation:
    def __init__(self, top: Topology, T):
        self.top = top
        self.T = T
        self.f = {}  # flow variables (tail, head, starttime, origin, destination)
        self.w = {}  # usage variables. Used to minimize number of linking variables
        self.edgelist = None

    def build(self, subproblems, g, edgelist):
        top, T = self.top, self.T
        f, w = self.f, self.w
        self.edgelist = edgelist

        # Create variables
        for edge in g.edges():
            v1, v2 = edge
            i, t = v1
            j, _ = v2

            allowed = top.dest_filter.get(i)
            for ec in top.ECs:
                o = ec.gpus[0]

                wvar = subproblems[o].addVar(name=f"w({i, j, t, o})")
                w[i, j, t, o] = wvar

                for d in top.gpus:
                    fvar = subproblems[o].addVar(name=f"f({i, j, t, o, d})")
                    if allowed is not None and d not in allowed:
                        fvar.ub = 0.0
                    f[i, j, t, o, d] = fvar
                    subproblems[o].addConstr(
                        w[i, j, t, o] >= f[i, j, t, o, d], f"xf {i, j, t, o, d}"
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
                            f[e1[0], e2[0], e1[1], o, d] for e1, e2 in g.out_edges(node)
                        )
                        - sum(
                            f[e1[0], e2[0], e1[1], o, d] for e1, e2 in g.in_edges(node)
                        )
                        == flow_dict[i, t, o, d],
                        f"flow {i, t, o, d}",
                    )

        for ec in top.ECs:
            o = ec.gpus[0]
            for node in g.nodes():
                i, t = node

                if i not in top.copy_nodes:
                    subproblems[o].addConstr(
                        sum(w[e1[0], e2[0], e1[1], o] for e1, e2 in g.out_edges(node))
                        == sum(
                            w[e1[0], e2[0], e1[1], o] for e1, e2 in g.in_edges(node)
                        ),
                        f"w pass through {i, t, o}",
                    )
                else:
                    in_total = sum(
                        w[e1[0], e2[0], e1[1], o] for e1, e2 in g.in_edges(node)
                    )
                    for oe1, oe2 in g.out_edges(node):
                        subproblems[o].addConstr(
                            in_total + min(flow_dict[i, t, o, d] for d in top.gpus)
                            >= w[oe1[0], oe2[0], oe1[1], o],
                            f"w stability {i, t, o, oe2[1]}",
                        )

    def priced_groups(self, o):
        return [[self.w[ei, ej, t, o] for ((ei, t), (ej, _)) in self.edgelist]]

    def f_values(self, o):
        return np.array(
            [
                self.f[ei, ej, t, o, d].X
                for ((ei, t), (ej, _)) in self.edgelist
                for d in self.top.gpus
            ]
        )


def all_gather_mirrored_dw(top: Topology, T, R, gurobi_env, cfg: Config):
    formulation = _AllGatherFormulation(top, T)
    m, w_arrays, f_arrays, g, elapsed, R = mirrored_dw(
        top,
        T,
        R,
        gurobi_env,
        formulation,
        name="all_gather",
        no_crossover=cfg.ag_no_crossover,
        phase1_alpha=cfg.ag_phase1_cost_weight,
        dw_tolerance=cfg.dw_tolerance,
        feasible_only=cfg.stop_at_feasible,
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
                f_sol[ei, ej, t, o, d] = array_sol[counter]
                counter += 1

        counter = 0
        for (ei, t), (ej, _) in edgelist:
            w_sol[ei, ej, t, o] = array_w_sol[counter]
            counter += 1
    return m, f_sol, w_sol, g, elapsed, R


def determine_throughput(top: Topology, gurobi_env, reverse=False):
    orient = (lambda a, b: (b, a)) if reverse else (lambda a, b: (a, b))
    G = nx.DiGraph()
    for edge, _ in top.edge_data.items():
        G.add_edge(*orient(*edge))
    for gpu in top.gpus:
        G.add_edge("s", gpu)

    limit = np.inf

    sinks = [ec.gpus[0] for ec in top.ECs] if top.ECs else top.gpus
    for gpu in sinks:
        f = {}
        m = gp.Model(env=gurobi_env)
        m.params.OutputFlag = 0

        x = m.addVar(name="x")

        for edge in G.edges():
            f[edge] = m.addVar(name=f"edge {edge}")
        for node in G.nodes():
            if node == "s":
                m.addConstr(
                    sum(f[edge] for edge in G.out_edges(node)) == len(top.gpus) * x
                )
            elif node == gpu:
                m.addConstr(
                    sum(f[edge] for edge in G.in_edges(node)) == len(top.gpus) * x
                )
            else:
                m.addConstr(
                    sum(f[edge] for edge in G.in_edges(node))
                    == sum(f[edge] for edge in G.out_edges(node))
                )
        for bc in top.bandwidth_constraints:
            m.addConstr(sum(f[orient(*edge)] for edge in bc.edges) <= bc.bound)
        for node in top.gpus:
            m.addConstr(f[("s", node)] <= x)
        m.setObjective(x, gp.GRB.MAXIMIZE)

        m.optimize()
        if m.status != gp.GRB.OPTIMAL:
            raise OptcclError(
                f"throughput LP for sink {gpu!r} did not solve to optimality "
                f"(Gurobi status {m.status}); cannot auto-determine a rate -- "
                f"pass -r explicitly"
            )
        limit = min(limit, m.objVal)
    if limit <= 0:
        raise OptcclError(
            "throughput LP found a zero rate (some GPU cannot send/receive any "
            "flow); check the topology's connectivity and bandwidth constraints"
        )
    return 1 / limit
