import logging

import networkx as nx
import gurobipy as gp
import numpy as np

from collections import defaultdict

from .topologies import Topology, ec_edge_preimage_maps
from .config import Config
from .dw import mirrored_dw
from .errors import OptcclError

logger = logging.getLogger(__name__)


class _AllToAllFormulation:
    def __init__(self, top: Topology, T):
        self.top = top
        self.T = T
        self.f = {}  # flow variables (tail, head, starttime, origin, destination)
        self.edgelist = None

    def build(self, subproblems, g, edgelist):
        top, T = self.top, self.T
        f = self.f
        self.edgelist = edgelist

        # Create variables
        for edge in g.edges():
            v1, v2 = edge
            i, t = v1
            j, _ = v2

            for ec in top.ECs:
                o = ec.gpus[0]

                for d in top.gpus:
                    fvar = subproblems[o].addVar(name=f"f({i, j, t, o, d})")
                    f[i, j, t, o, d] = fvar

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

    def priced_groups(self, o):
        return [
            [self.f[ei, ej, t, o, d] for ((ei, t), (ej, _)) in self.edgelist]
            for d in self.top.gpus
        ]

    def f_values(self, o):
        return np.zeros(0)


def all_to_all_mirrored_dw(top: Topology, T, R, gurobi_env, cfg: Config):
    formulation = _AllToAllFormulation(top, T)
    m, w_arrays, _f_arrays, g, elapsed, R = mirrored_dw(
        top,
        T,
        R,
        gurobi_env,
        formulation,
        name="all_to_all",
        no_crossover=cfg.aa_no_crossover,
        phase1_alpha=cfg.aa_phase1_cost_weight,
        feasible_only=cfg.stop_at_feasible,
        dw_tolerance=cfg.dw_tolerance,
        rate_retry_max=cfg.rate_retry_max,
        rate_retry_factor=cfg.rate_retry_factor,
    )
    edgelist = formulation.edgelist

    f_sol = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        if o not in w_arrays:
            continue
        array_sol = w_arrays[o]
        counter = 0
        for d in top.gpus:
            for (ei, t), (ej, _) in edgelist:
                f_sol[ei, ej, t, o, d] = array_sol[counter]
                counter += 1

    return m, f_sol, g, elapsed, R


def determine_throughput(top: Topology, gurobi_env):
    G = nx.DiGraph()
    for edge, _ in top.edge_data.items():
        G.add_edge(edge[0], edge[1])

    gpu_set = set(top.gpus)
    N = len(top.gpus)

    if top.ECs:
        origins = [ec.gpus[0] for ec in top.ECs]
        preimages = ec_edge_preimage_maps(top.ECs, G.edges())
    else:
        identity = {e: e for e in G.edges()}
        origins = list(top.gpus)
        preimages = {o: [identity] for o in origins}

    f = {}  # (edge, origin) -> flow var
    m = gp.Model(env=gurobi_env)
    m.params.OutputFlag = 0
    x = m.addVar(name="x")
    for o in origins:
        for edge in G.edges():
            f[edge, o] = m.addVar(name=f"edge {edge, o}")
        for node in G.nodes():
            outflow = sum(f[edge, o] for edge in G.out_edges(node))
            inflow = sum(f[edge, o] for edge in G.in_edges(node))
            if node == o:
                m.addLConstr(outflow - inflow == (N - 1) * x)
            elif node in gpu_set:
                m.addLConstr(outflow - inflow == -x)
            else:
                m.addLConstr(outflow - inflow == 0)

    for bc in top.bandwidth_constraints:
        m.addConstr(
            sum(
                f[inv[edge], o]
                for edge in bc.edges
                for o in origins
                for inv in preimages[o]
            )
            <= bc.bound
        )
    m.setObjective(x, gp.GRB.MAXIMIZE)

    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        raise OptcclError(
            f"all_to_all throughput LP did not solve to optimality (Gurobi "
            f"status {m.status}); cannot auto-determine a rate -- pass -r explicitly"
        )
    if m.objVal <= 0:
        raise OptcclError(
            "all_to_all throughput LP found a zero rate (some GPU cannot "
            "send/receive any flow); check the topology's connectivity and "
            "bandwidth constraints"
        )
    return 1 / m.objVal
