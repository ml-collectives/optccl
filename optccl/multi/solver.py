import logging
from collections import defaultdict

import gurobipy as gp
import networkx as nx
import numpy as np

from ..dw import mirrored_dw
from ..errors import OptcclError
from ..topologies import ec_edge_preimage_maps

logger = logging.getLogger(__name__)


class _MultiCollectiveFormulation:
    def __init__(self, top, T):
        self.top = top
        self.T = T
        self.f = {}
        self.w = {}
        self.p2p_dests = defaultdict(set)
        self.edgelist = None

    def build(self, subproblems, g, edgelist):
        top, T = self.top, self.T
        f, w, p2p_dests = self.f, self.w, self.p2p_dests
        self.edgelist = edgelist

        for edge in g.edges():
            v1, v2 = edge
            i, t = v1
            j, _ = v2

            for ec in top.ECs:
                o = ec.gpus[0]

                for p2p in top.p2p_by_source[o]:
                    p2p_dests[o].add(p2p.sink)
                for dest in p2p_dests[o]:
                    w[i, j, t, o, dest, "p2p"] = subproblems[o].addVar(
                        name=f"wp2p({i, j, t, o, dest})"
                    )

                for idx, gather in enumerate(top.gather_by_source[o]):
                    w[i, j, t, o, "g", idx] = subproblems[o].addVar(
                        name=f"wg{idx}({i, j, t, o})"
                    )
                    for d in gather.sinks:
                        fvar = subproblems[o].addVar(name=f"fg{idx}({i, j, t, o, d})")
                        f[i, j, t, o, d, "g", idx] = fvar
                        subproblems[o].addConstr(
                            w[i, j, t, o, "g", idx] >= fvar, f"xf {i, j, t, o, d}"
                        )

                for idx, reduce in enumerate(top.reduce_by_sink[o]):
                    w[i, j, t, o, "r", idx] = subproblems[o].addVar(
                        name=f"wr{idx}({i, j, t, o})"
                    )
                    for d in reduce.sources:
                        fvar = subproblems[o].addVar(name=f"fr{idx}({i, j, t, o, d})")
                        f[i, j, t, o, d, "r", idx] = fvar
                        subproblems[o].addConstr(
                            w[i, j, t, o, "r", idx] >= fvar, f"xf {i, j, t, o, d}"
                        )

        p2p_flow_dict = defaultdict(int)
        for p2p in top.p2ps:
            p2p_flow_dict[p2p.source, 0, p2p.source, p2p.sink] += p2p.demand
            p2p_flow_dict[p2p.sink, T, p2p.source, p2p.sink] -= p2p.demand

        gather_flow_dict = defaultdict(int)
        for o in top.effective_gpus:
            for idx, gather in enumerate(top.gather_by_source[o]):
                for d in gather.sinks:
                    gather_flow_dict[gather.source, 0, gather.source, d, idx] += (
                        gather.demand
                    )
                    gather_flow_dict[d, T, gather.source, d, idx] -= gather.demand

        reduce_flow_dict = defaultdict(int)
        for o in top.effective_gpus:
            for idx, reduce in enumerate(top.reduce_by_sink[o]):
                for d in reduce.sources:
                    reduce_flow_dict[reduce.sink, T, reduce.sink, d, idx] -= (
                        reduce.demand
                    )
                    reduce_flow_dict[d, 0, reduce.sink, d, idx] += reduce.demand

        for ec in top.ECs:
            o = ec.gpus[0]
            for d in p2p_dests[o]:
                for node in g.nodes():
                    i, t = node
                    subproblems[o].addConstr(
                        sum(
                            w[e1[0], e2[0], e1[1], o, d, "p2p"]
                            for e1, e2 in g.out_edges(node)
                        )
                        - sum(
                            w[e1[0], e2[0], e1[1], o, d, "p2p"]
                            for e1, e2 in g.in_edges(node)
                        )
                        == p2p_flow_dict[i, t, o, d],
                        f"flow {i, t, o, d}",
                    )

        for ec in top.ECs:
            o = ec.gpus[0]
            for idx, gather in enumerate(top.gather_by_source[o]):
                for d in gather.sinks:
                    for node in g.nodes():
                        i, t = node
                        subproblems[o].addConstr(
                            sum(
                                f[e1[0], e2[0], e1[1], o, d, "g", idx]
                                for e1, e2 in g.out_edges(node)
                            )
                            - sum(
                                f[e1[0], e2[0], e1[1], o, d, "g", idx]
                                for e1, e2 in g.in_edges(node)
                            )
                            == gather_flow_dict[i, t, o, d, idx],
                            f"flow {i, t, o, d}",
                        )

        for ec in top.ECs:
            o = ec.gpus[0]
            for idx, gather in enumerate(top.gather_by_source[o]):
                for node in g.nodes():
                    i, t = node
                    if i not in top.copy_nodes:
                        subproblems[o].addConstr(
                            sum(
                                w[e1[0], e2[0], e1[1], o, "g", idx]
                                for e1, e2 in g.out_edges(node)
                            )
                            == sum(
                                w[e1[0], e2[0], e1[1], o, "g", idx]
                                for e1, e2 in g.in_edges(node)
                            ),
                            f"w pass through {i, t, o}",
                        )
                    else:
                        in_total = sum(
                            w[e1[0], e2[0], e1[1], o, "g", idx]
                            for e1, e2 in g.in_edges(node)
                        )
                        for oe1, oe2 in g.out_edges(node):
                            subproblems[o].addConstr(
                                in_total
                                + min(
                                    gather_flow_dict[i, t, o, d, idx]
                                    for d in gather.sinks
                                )
                                >= w[oe1[0], oe2[0], oe1[1], o, "g", idx],
                                f"w stability {i, t, o, oe2[1]}",
                            )

        for ec in top.ECs:
            o = ec.gpus[0]
            for idx, reduce in enumerate(top.reduce_by_sink[o]):
                for d in reduce.sources:
                    for node in g.nodes():
                        i, t = node
                        subproblems[o].addConstr(
                            sum(
                                f[e1[0], e2[0], e1[1], o, d, "r", idx]
                                for e1, e2 in g.out_edges(node)
                            )
                            - sum(
                                f[e1[0], e2[0], e1[1], o, d, "r", idx]
                                for e1, e2 in g.in_edges(node)
                            )
                            == reduce_flow_dict[i, t, o, d, idx],
                            f"flow {i, t, o, d}",
                        )

        for ec in top.ECs:
            o = ec.gpus[0]
            for idx, reduce in enumerate(top.reduce_by_sink[o]):
                for node in g.nodes():
                    i, t = node
                    if i not in top.reduce_nodes:
                        subproblems[o].addConstr(
                            sum(
                                w[e1[0], e2[0], e1[1], o, "r", idx]
                                for e1, e2 in g.out_edges(node)
                            )
                            == sum(
                                w[e1[0], e2[0], e1[1], o, "r", idx]
                                for e1, e2 in g.in_edges(node)
                            ),
                            f"w pass through {i, t, o}",
                        )
                    else:
                        out_total = sum(
                            w[e1[0], e2[0], e1[1], o, "r", idx]
                            for e1, e2 in g.out_edges(node)
                        )
                        for ie1, ie2 in g.in_edges(node):
                            subproblems[o].addConstr(
                                out_total
                                - min(
                                    reduce_flow_dict[i, t, o, d, idx]
                                    for d in reduce.sources
                                )
                                >= w[ie1[0], ie2[0], ie1[1], o, "r", idx],
                                f"w stability {i, t, o, ie2[1]}",
                            )

    def priced_groups(self, o):
        edgelist, w = self.edgelist, self.w
        groups = [
            [w[ei, ej, t, o, d, "p2p"] for ((ei, t), (ej, _)) in edgelist]
            for d in self.p2p_dests[o]
        ]
        groups += [
            [w[ei, ej, t, o, "g", idx] for ((ei, t), (ej, _)) in edgelist]
            for idx in range(len(self.top.gather_by_source[o]))
        ]
        groups += [
            [w[ei, ej, t, o, "r", idx] for ((ei, t), (ej, _)) in edgelist]
            for idx in range(len(self.top.reduce_by_sink[o]))
        ]
        return groups

    def f_values(self, o):
        edgelist, f, w = self.edgelist, self.f, self.w
        # p2p carries no separate per-participant flows, so its w values stand
        # in for them -- mirrors the w concatenation's p2p block.
        arrs = [
            np.array([w[ei, ej, t, o, d, "p2p"].X for ((ei, t), (ej, _)) in edgelist])
            for d in self.p2p_dests[o]
        ]
        arrs += [
            np.array(
                [f[ei, ej, t, o, d, "g", idx].X for ((ei, t), (ej, _)) in edgelist]
            )
            for idx, gather in enumerate(self.top.gather_by_source[o])
            for d in gather.sinks
        ]
        arrs += [
            np.array(
                [f[ei, ej, t, o, d, "r", idx].X for ((ei, t), (ej, _)) in edgelist]
            )
            for idx, reduce in enumerate(self.top.reduce_by_sink[o])
            for d in reduce.sources
        ]
        return np.concatenate(arrs) if arrs else np.zeros(0)


def multiple_collective_generic_mirrored_dw(
    top,
    T,
    R,
    gurobi_env,
    no_crossover=True,
    phase1_cost_weight_alpha=0.01,
    dw_tolerance=1e-7,
    feasible_only=False,
    rate_retry_max=0,
    rate_retry_factor=1.01,
):
    formulation = _MultiCollectiveFormulation(top, T)
    m, w_arrays, f_arrays, g, elapsed, R = mirrored_dw(
        top,
        T,
        R,
        gurobi_env,
        formulation,
        name="multi",
        no_crossover=no_crossover,
        phase1_alpha=phase1_cost_weight_alpha,
        dw_tolerance=dw_tolerance,
        feasible_only=feasible_only,
        rate_retry_max=rate_retry_max,
        rate_retry_factor=rate_retry_factor,
    )
    edgelist = formulation.edgelist
    gsize = len(edgelist)
    p2p_dests = formulation.p2p_dests

    f_sol = {}
    w_sol = {}
    edge_utilization = defaultdict(float)
    for ec in top.ECs:
        o = ec.gpus[0]
        if o not in w_arrays:
            continue
        array_f = f_arrays[o]
        array_w = w_arrays[o]

        counter_f = 0
        for d in p2p_dests[o]:
            for (ei, t), (ej, _) in edgelist:
                f_sol[ei, ej, t, o, d, "p2p"] = array_f[counter_f]
                counter_f += 1

        for idx, gather in enumerate(top.gather_by_source[o]):
            for d in gather.sinks:
                for (ei, t), (ej, _) in edgelist:
                    f_sol[ei, ej, t, o, d, "g", idx] = array_f[counter_f]
                    counter_f += 1

        for idx, reduce in enumerate(top.reduce_by_sink[o]):
            for d in reduce.sources:
                for (ei, t), (ej, _) in edgelist:
                    f_sol[ei, ej, t, o, d, "r", idx] = array_f[counter_f]
                    counter_f += 1

        # array_w layout per origin: p2p sub-arrays (one per dest) then gather
        # (per idx) then reduce (per idx), each of length gsize -- mirrors the
        # priced-group concatenation order.
        counter_w = 0
        for d in p2p_dests[o]:
            for (ei, t), (ej, _) in edgelist:
                w_sol[ei, ej, t, o, d, "p2p"] = array_w[counter_w]
                counter_w += 1
        for idx in range(len(top.gather_by_source[o])):
            for (ei, t), (ej, _) in edgelist:
                w_sol[ei, ej, t, o, "g", idx] = array_w[counter_w]
                counter_w += 1
        for idx in range(len(top.reduce_by_sink[o])):
            for (ei, t), (ej, _) in edgelist:
                w_sol[ei, ej, t, o, "r", idx] = array_w[counter_w]
                counter_w += 1

        # Per-physical-edge utilization summed over time, demands, and EC shifts.
        n_groups = (
            len(p2p_dests[o])
            + len(top.gather_by_source[o])
            + len(top.reduce_by_sink[o])
        )
        if n_groups == 0:
            continue
        w_per_edge = array_w.reshape(n_groups, gsize).sum(axis=0)
        for edge_idx, ((ei, t1), (ej, _)) in enumerate(edgelist):
            for shift in range(len(ec.gpus)):
                mapped_e = (ec.shift_fn(ei, shift), ec.shift_fn(ej, shift))
                edge_utilization[mapped_e] += w_per_edge[edge_idx]

    return m, f_sol, w_sol, dict(edge_utilization), elapsed, R


def _ec_reduce(keys, rep_of):
    if rep_of is None:
        return keys
    keyset = set(keys)
    kept = [k for k in keys if rep_of.get(k, k) == k]
    assert all(rep_of[k] in keyset for k in keyset - set(kept)), (
        "EC member has a demand its representative lacks"
    )
    return kept


def _add_aa_flow(m, G, top, x, rep_of=None):
    demand = defaultdict(int)
    origins = set()
    for p2p in top.p2ps:
        if p2p.source == p2p.sink:
            continue
        demand[p2p.source, p2p.sink] += p2p.demand
        origins.add(p2p.source)
    origins = _ec_reduce(sorted(origins, key=str), rep_of)

    f_p2p = {}
    for o in origins:
        dest_dem = {d: dem for (oo, d), dem in demand.items() if oo == o}
        total_out = sum(dest_dem.values())
        for edge in G.edges():
            f_p2p[edge, o] = m.addVar(name=f"p2p {edge, o}")
        for node in G.nodes():
            outflow = sum(f_p2p[e, o] for e in G.out_edges(node))
            inflow = sum(f_p2p[e, o] for e in G.in_edges(node))
            if node == o:
                m.addLConstr(outflow - inflow == total_out * x)
            elif node in dest_dem:
                m.addLConstr(outflow - inflow == -dest_dem[node] * x)
            else:
                m.addLConstr(outflow - inflow == 0)
    return f_p2p, origins


def _add_ag_flow(m, G, top, x, rep_of=None):
    sources = _ec_reduce(sorted(top.gather_by_source.keys(), key=str), rep_of)
    f_g = {}
    for s in sources:
        gathers = top.gather_by_source[s]
        total = sum(g.demand * len(g.sinks) for g in gathers)
        by_sink = defaultdict(int)
        for g in gathers:
            for d in g.sinks:
                by_sink[d] += g.demand
        for edge in G.edges():
            f_g[edge, s] = m.addVar(name=f"g {edge, s}")
        sink_vars = {d: m.addVar(name=f"gsink {s, d}") for d in by_sink}
        for node in G.nodes():
            outflow = sum(f_g[e, s] for e in G.out_edges(node))
            inflow = sum(f_g[e, s] for e in G.in_edges(node))
            extra_out = sink_vars[node] if node in sink_vars else 0
            if node == s:
                m.addLConstr(outflow + extra_out - inflow == total * x)
            else:
                m.addLConstr(outflow + extra_out - inflow == 0)
        for d, v in sink_vars.items():
            m.addLConstr(v <= by_sink[d] * x)
    return f_g, sources


def _add_rs_flow(m, G, top, x, rep_of=None):
    sinks = _ec_reduce(sorted(top.reduce_by_sink.keys(), key=str), rep_of)
    f_r = {}
    for k in sinks:
        reduces = top.reduce_by_sink[k]
        total = sum(r.demand * len(r.sources) for r in reduces)
        by_source = defaultdict(int)
        for r in reduces:
            for d in r.sources:
                by_source[d] += r.demand
        for edge in G.edges():
            f_r[edge, k] = m.addVar(name=f"r {edge, k}")
        source_vars = {d: m.addVar(name=f"rsrc {k, d}") for d in by_source}
        for node in G.nodes():
            outflow = sum(f_r[e, k] for e in G.out_edges(node))
            inflow = sum(f_r[e, k] for e in G.in_edges(node))
            extra_in = source_vars[node] if node in source_vars else 0
            if node == k:
                m.addLConstr(outflow - inflow - extra_in == -total * x)
            else:
                m.addLConstr(outflow - inflow - extra_in == 0)
        for d, v in source_vars.items():
            m.addLConstr(v <= by_source[d] * x)
    return f_r, sinks


def determine_throughput(top, gurobi_env):
    G = nx.DiGraph()
    for edge in top.edge_data.keys():
        G.add_edge(edge[0], edge[1])

    identity = [{e: e for e in G.edges()}]
    if top.ECs:
        rep_of = {g: ec.gpus[0] for ec in top.ECs for g in ec.gpus}
        preimages = ec_edge_preimage_maps(top.ECs, G.edges())
    else:
        rep_of = None
        preimages = {}

    m = gp.Model(env=gurobi_env)
    m.params.OutputFlag = 0
    x = m.addVar(name="x")

    families = []
    if top.p2ps:
        families.append(_add_aa_flow(m, G, top, x, rep_of))
    if top.gathers:
        families.append(_add_ag_flow(m, G, top, x, rep_of))
    if top.reduces:
        families.append(_add_rs_flow(m, G, top, x, rep_of))
    assert families, "workload has no demands"

    for bc in top.bandwidth_constraints:
        m.addConstr(
            sum(
                fv[inv[e], o]
                for fv, keys in families
                for o in keys
                for inv in preimages.get(o, identity)
                for e in bc.edges
            )
            <= bc.bound
        )

    m.setObjective(x, gp.GRB.MAXIMIZE)
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        raise OptcclError(
            f"multi-collective throughput LP did not solve to optimality "
            f"(Gurobi status {m.status}); cannot auto-determine a rate -- "
            f"pass -r explicitly"
        )
    if m.objVal <= 0:
        raise OptcclError(
            "multi-collective throughput LP found a zero rate (some demand "
            "endpoint cannot send/receive any flow); check the workload's "
            "topology connectivity and bandwidth constraints"
        )
    return 1.0 / m.objVal
