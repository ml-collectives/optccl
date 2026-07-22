import logging
import time
from collections import defaultdict
import multiprocessing as mp

import networkx as nx
import numpy as np
import gurobipy as gp

from .topologies import Topology

logger = logging.getLogger(__name__)


class IPath:
    __slots__ = ("branch_nodes", "suffix_ids", "suffix_nodes", "dest")

    def __init__(self, path, all_branch_nodes, eidx):
        self.branch_nodes = []
        self.dest = path[-1]
        self.suffix_ids = {}
        self.suffix_nodes = {}

        total_suffix_set = set(zip(path[:-1], path[1:]))
        for i, node in enumerate(path):
            if node in all_branch_nodes:
                self.branch_nodes.append(node)
                self.suffix_ids[node] = [eidx[e] for e in total_suffix_set if e in eidx]
            if not total_suffix_set:
                break
            total_suffix_set.remove((path[i], path[i + 1]))
        for bn in self.branch_nodes:
            try:
                idx = path.index(bn)
                self.suffix_nodes[bn] = path[idx:]
            except ValueError:
                self.suffix_nodes[bn] = []

    def get_cost_fast(self, cost_arr, active_branch_nodes):
        for bn in reversed(self.branch_nodes):
            if bn in active_branch_nodes:
                ids = self.suffix_ids[bn]
                if not ids:
                    return 0.0
                total = 0.0
                for i in ids:
                    total += cost_arr[i]
                return total
        return 1e300

    def get_active_suffix(self, active_branch_nodes):
        for bn in reversed(self.branch_nodes):
            if bn in active_branch_nodes:
                return self.suffix_ids[bn]
        return None

    def get_active_suffix_nodes(self, active_branch_nodes):
        for bn in reversed(self.branch_nodes):
            if bn in active_branch_nodes:
                return self.suffix_nodes.get(bn)
        return None


class ISTree:
    __slots__ = (
        "n_edges",
        "mult",
        "destinations",
        "active_branch_nodes",
        "_fp",
        "dest_suffixes",
        "dest_full_edges",
        "parent",
        "origin",
        "eidx",
    )

    def __init__(self, origin, n_edges, eidx=None):
        self.n_edges = n_edges
        self.mult = [0] * n_edges
        self.destinations = set()
        self.active_branch_nodes = {origin}
        self._fp = None
        self.dest_suffixes = {}  # dest -> list of edge ids used
        self.dest_full_edges = {}  # dest -> list of edge ids (full path in tree)
        self.parent = {}
        self.origin = origin
        self.eidx = eidx

    def _walk_to_origin(self, node):
        edges = []
        while node != self.origin:
            entry = self.parent.get(node)
            if entry is None:
                break
            par, eid = entry
            if eid is not None:
                edges.append(eid)
            node = par
        edges.reverse()
        return edges

    def add_path(self, ipath):
        self.destinations.add(ipath.dest)
        self._fp = None
        ids = ipath.get_active_suffix(self.active_branch_nodes)
        self.dest_suffixes[ipath.dest] = ids if ids else []

        suffix_nodes = ipath.get_active_suffix_nodes(self.active_branch_nodes)
        branch_node = None
        suffix_node_set = set()
        if suffix_nodes and len(suffix_nodes) >= 2:
            branch_node = suffix_nodes[0]
            suffix_node_set = set(suffix_nodes)
            for u, v in zip(suffix_nodes[:-1], suffix_nodes[1:]):
                if v not in self.parent:
                    eid = self.eidx.get((u, v)) if self.eidx else None
                    self.parent[v] = (u, eid)

        if branch_node is not None:
            trunk_edges = self._walk_to_origin(branch_node)
            suffix_edge_ids = ids if ids else []
            self.dest_full_edges[ipath.dest] = trunk_edges + suffix_edge_ids
        else:
            self.dest_full_edges[ipath.dest] = []

        if ids:
            for i in ids:
                self.mult[i] += 1
        for bn in ipath.branch_nodes:
            if bn in suffix_node_set:
                self.active_branch_nodes.add(bn)

    def total_multiplicity(self):
        return sum(self.mult)

    def reduced_cost_arr(self, pi_arr):
        total = 0.0
        for i, m in enumerate(self.mult):
            if m > 0:
                total += pi_arr[i] * m
        return total

    def fingerprint(self):
        if self._fp is None:
            parts = []
            for i, m in enumerate(self.mult):
                if m > 0:
                    parts.append((i, m))
            self._fp = hash(tuple(parts))
        return self._fp

    def nonzero_edges(self):
        for i, m in enumerate(self.mult):
            if m > 0:
                yield i, m


def determine_branching(top: Topology, g, collective):
    branching_nodes = set()

    base_branch = set()
    if collective in ("all_gather", "all_reduce"):
        # fan-out (copy) trees may branch at any copy-capable node
        base_branch.update(top.copy_nodes)
    elif collective == "reduce_scatter":
        # fan-in (reduce) trees may branch at any reduce-capable node
        base_branch.update(top.reduce_nodes)
    for node in g.nodes():
        i, _ = node
        if i in base_branch:
            branching_nodes.add(node)
    return branching_nodes


def _join_path(w, pred_p, succ_p):
    path = []
    x = w
    while x is not None:
        path.append(x)
        x = pred_p[x]
    path.reverse()
    x = succ_p[w]
    while x is not None:
        path.append(x)
        x = succ_p[x]
    return path


def _bidirectional_path(succ, pred, source, target):
    if source == target:
        return [source]
    if source not in succ or target not in pred:
        return None
    pred_p = {source: None}
    succ_p = {target: None}
    fwd = [source]
    bwd = [target]
    while fwd and bwd:
        if len(fwd) <= len(bwd):
            this_level = fwd
            fwd = []
            for u in this_level:
                for v in succ.get(u, ()):
                    if v not in pred_p:
                        pred_p[v] = u
                        fwd.append(v)
                    if v in succ_p:
                        return _join_path(v, pred_p, succ_p)
        else:
            this_level = bwd
            bwd = []
            for u in this_level:
                for v in pred.get(u, ()):
                    if v not in succ_p:
                        succ_p[v] = u
                        bwd.append(v)
                    if v in pred_p:
                        return _join_path(v, pred_p, succ_p)
    return None


def _decompose_commodity(succ_od, source, sink, branching_nodes, eidx, tolerance):
    pred_od = {}
    for u, nbrs in succ_od.items():
        for v, fl in nbrs.items():
            pv = pred_od.get(v)
            if pv is None:
                pv = pred_od[v] = {}
            pv[u] = fl

    paths = []
    while True:
        path = _bidirectional_path(succ_od, pred_od, source, sink)
        if path is None:
            break
        path_flow = min(succ_od[path[k]][path[k + 1]] for k in range(len(path) - 1))
        paths.append(IPath(path, branching_nodes, eidx))
        for k in range(len(path) - 1):
            u = path[k]
            v = path[k + 1]
            nf = succ_od[u][v] - path_flow
            if nf < tolerance:
                del succ_od[u][v]
                del pred_od[v][u]
            else:
                succ_od[u][v] = nf
                pred_od[v][u] = nf
    return paths


# Module-level globals
_PD_BRANCHING = None
_PD_EIDX = None
_PD_T = None
_PD_TOL = None


def _pd_init_worker(branching_nodes, eidx, T, tolerance):
    global _PD_BRANCHING, _PD_EIDX, _PD_T, _PD_TOL
    _PD_BRANCHING = branching_nodes
    _PD_EIDX = eidx
    _PD_T = T
    _PD_TOL = tolerance


def _pd_worker(arg):
    od, succ_od = arg
    o, d = od
    return od, _decompose_commodity(
        succ_od, (o, 0), (d, _PD_T), _PD_BRANCHING, _PD_EIDX, _PD_TOL
    )


def path_decomposition(g, f, branching_nodes, eidx, cfg):
    """Decompose the LP flow ``f`` into per-commodity source->sink paths (``IPath``s).

    Each commodity ``(o, d)`` is an independent shortest-path flow decomposition. We build
    plain nested-dict adjacency (no networkx ``DiGraph``) in ``g.edges()`` order and run a
    pure-Python bidirectional BFS, which reproduces the old nx-based path choices exactly
    while dropping nx's per-call overhead. Commodities are independent, so when
    ``cfg.n_workers`` allows and there are enough of them we fan them out across processes.
    """
    tolerance = cfg.decomposition_tolerance

    by_edge = defaultdict(list)
    all_dests = set()
    for (i, j, t, o, d), fval in f.items():
        all_dests.add(d)
        if fval > tolerance:
            by_edge[(i, j, t)].append((o, d, fval))

    # Build per-commodity forward adjacency
    succ = {}
    T = -1
    for (i, t), (j, t2) in g.edges():
        if t2 > T:
            T = t2
        lst = by_edge.get((i, j, t))
        if not lst:
            continue
        u = (i, t)
        v = (j, t2)
        for o, d, fval in lst:
            od = (o, d)
            s = succ.get(od)
            if s is None:
                s = succ[od] = {}
            su = s.get(u)
            if su is None:
                su = s[u] = {}
            su[v] = fval

    dest_set = {(d, T) for d in all_dests}

    n_workers = mp.cpu_count() if cfg.n_workers == -1 else cfg.n_workers
    commodities = list(succ.items())

    f_paths = defaultdict(list)
    if n_workers > 1 and len(commodities) >= max(2, n_workers):
        chunksize = max(1, len(commodities) // (n_workers * 4))
        with mp.Pool(
            n_workers,
            initializer=_pd_init_worker,
            initargs=(branching_nodes, eidx, T, tolerance),
        ) as pool:
            for od, paths in pool.imap_unordered(_pd_worker, commodities, chunksize):
                f_paths[od] = paths
    else:
        for od, succ_od in commodities:
            o, d = od
            f_paths[od] = _decompose_commodity(
                succ_od, (o, 0), (d, T), branching_nodes, eidx, tolerance
            )

    return f_paths, list(dest_set)


def _build_one_tree(
    origin,
    dest_order,
    dest_ipaths,
    cost_arr,
    n_edges,
    selector,
    rng_np,
    sample_k=60,
    eidx=None,
):
    tree = ISTree(origin, n_edges, eidx=eidx)
    for d in dest_order:
        candidates = dest_ipaths.get(d)
        if not candidates:
            continue

        if len(candidates) > sample_k and selector != "random":
            idxs = rng_np.choice(len(candidates), size=sample_k, replace=False)
            subset = [candidates[i] for i in idxs]
        else:
            subset = candidates

        if selector == "best":
            best_cost = float("inf")
            best_path = None
            for ip in subset:
                c = ip.get_cost_fast(cost_arr, tree.active_branch_nodes)
                if c < best_cost:
                    best_cost = c
                    best_path = ip
            if best_path is not None:
                tree.add_path(best_path)

        elif selector == "sample":
            costs = [
                ip.get_cost_fast(cost_arr, tree.active_branch_nodes) for ip in subset
            ]
            min_c = min(costs)
            if min_c >= 1e200:
                tree.add_path(subset[rng_np.randint(0, len(subset))])
            else:
                scale = max(abs(min_c), 1e-6)
                weights = [
                    np.exp(-(c - min_c) / scale) if c < 1e200 else 0.0 for c in costs
                ]
                total = sum(weights)
                if total > 0:
                    r = rng_np.random() * total
                    cumul = 0.0
                    for j, w in enumerate(weights):
                        cumul += w
                        if cumul >= r:
                            tree.add_path(subset[j])
                            break
                    else:
                        tree.add_path(subset[-1])
                else:
                    tree.add_path(subset[rng_np.randint(0, len(subset))])

        elif selector == "random":
            tree.add_path(candidates[rng_np.randint(0, len(candidates))])

    return tree


# Module-level globals set by _init_worker for each subprocess
_W_ORIGIN = None
_W_DESTS = None
_W_DEST_IPATHS = None
_W_N_EDGES = None
_W_EIDX = None


def _init_worker(origin, dests, dest_ipaths, n_edges, eidx):
    global _W_ORIGIN, _W_DESTS, _W_DEST_IPATHS, _W_N_EDGES, _W_EIDX
    _W_ORIGIN = origin
    _W_DESTS = dests
    _W_DEST_IPATHS = dest_ipaths
    _W_N_EDGES = n_edges
    _W_EIDX = eidx


def _worker_build_batch(args):
    batch_costs, batch_sels, batch_orders, seed, sample_k, pi_for_rc = args
    rng_np = np.random.RandomState(seed)

    results = []
    seen = set()
    n_rc_rej = 0

    for ca, sel, order in zip(batch_costs, batch_sels, batch_orders):
        tree = _build_one_tree(
            _W_ORIGIN,
            order,
            _W_DEST_IPATHS,
            ca,
            _W_N_EDGES,
            sel,
            rng_np,
            sample_k,
            eidx=_W_EIDX,
        )
        if len(tree.destinations) < len(_W_DESTS):
            continue
        if pi_for_rc is not None:
            rc = tree.reduced_cost_arr(pi_for_rc)
            if rc >= 1.0 - 1e-8:
                n_rc_rej += 1
                continue
        fp = tree.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        results.append(
            (
                tree.mult,
                tree.destinations,
                fp,
                tree.dest_suffixes,
                tree.parent,
                tree.dest_full_edges,
            )
        )

    return results, n_rc_rej


class MasterLP:
    def __init__(self, bc_coef, rhs, edge_list, gurobi_env):
        self.edge_list = edge_list
        self.n_edges = len(edge_list)
        self.bc_coef = bc_coef
        self.n_bc = bc_coef.shape[1]
        self.model = gp.Model("master", env=gurobi_env)
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("Method", 2)
        self.model.setAttr("ModelSense", gp.GRB.MAXIMIZE)
        self.constrs = []
        for b in range(self.n_bc):
            self.constrs.append(self.model.addConstr(gp.LinExpr() <= rhs[b]))
        self.trees = []
        self.lam = []
        self.model.update()

    def add_trees(self, new_trees):
        for tree in new_trees:
            load = np.zeros(self.n_bc)
            for i, m in tree.nonzero_edges():
                load += m * self.bc_coef[i]
            col = gp.Column()
            for b in range(self.n_bc):
                if load[b] != 0.0:
                    col.addTerms(load[b], self.constrs[b])
            var = self.model.addVar(lb=0.0, obj=1.0, column=col)
            self.trees.append(tree)
            self.lam.append(var)
        self.model.update()

    def solve(self):
        self.model.optimize()
        return self.model.status == gp.GRB.OPTIMAL

    def get_objective(self):
        return self.model.ObjVal

    def get_duals_arr(self):
        return [c.Pi for c in self.constrs]

    def get_solution(self, edge_list):
        sol = []
        for i, tree in enumerate(self.trees):
            vol = self.lam[i].X
            if vol > 1e-10:
                edge_dict = {}
                for eid, m in tree.nonzero_edges():
                    edge_dict[edge_list[eid]] = m
                suffixes = {}
                for dest, ids in tree.dest_suffixes.items():
                    suffixes[dest] = [edge_list[eid] for eid in ids]
                # Compute full paths
                dest_order = list(tree.dest_suffixes.keys())
                dest_full_eids = {}
                for d in dest_order:
                    dest_full_eids[d] = set(tree.dest_full_edges.get(d, []))

                dest_suffix_set = {
                    d: set(tree.dest_suffixes.get(d, [])) for d in dest_order
                }
                edge_copy_dests = {}
                for eid, m in tree.nonzero_edges():
                    e = edge_list[eid]
                    copies = []
                    for d in dest_order:
                        if eid not in dest_full_eids[d]:
                            continue
                        if eid in dest_suffix_set[d]:
                            copies.append({d[0]})
                        else:
                            if copies:
                                copies[-1].add(d[0])
                    edge_copy_dests[e] = copies
                sol.append((edge_dict, vol, suffixes, edge_copy_dests))
        return sol

    def n_active(self):
        return sum(1 for v in self.lam if v.X > 1e-10)

    def prune_inactive(self, rng_np, keep_fraction=0.1):
        inactive = [i for i in range(len(self.trees)) if self.lam[i].X <= 1e-10]
        if not inactive:
            return 0
        n_keep = int(len(inactive) * keep_fraction)
        rng_np.shuffle(inactive)
        to_remove = set(inactive[n_keep:])
        if not to_remove:
            return 0
        self.model.remove([self.lam[i] for i in to_remove])
        self.model.update()
        new_trees, new_lam = [], []
        for i in range(len(self.trees)):
            if i not in to_remove:
                new_trees.append(self.trees[i])
                new_lam.append(self.lam[i])
        self.trees = new_trees
        self.lam = new_lam
        return len(to_remove)


def _reconstruct_tree(
    mult_list,
    dest_set,
    origin,
    n_edges,
    dest_suffixes=None,
    parent=None,
    dest_full_edges=None,
):
    tree = ISTree.__new__(ISTree)
    tree.n_edges = n_edges
    tree.mult = mult_list
    tree.destinations = dest_set
    tree.active_branch_nodes = set()  # not needed post-build
    tree._fp = None
    tree.dest_suffixes = dest_suffixes if dest_suffixes is not None else {}
    tree.dest_full_edges = dest_full_edges if dest_full_edges is not None else {}
    tree.parent = parent if parent is not None else {}
    tree.origin = origin
    tree.eidx = None  # not needed post-build
    return tree


def pool_decompose(
    origin,
    dests,
    f_paths,
    w,
    eidx,
    gurobi_env,
    cfg,
    bc_coef,
    bc_bounds,
    R,
    pooled_w_rhs,
    bc_rhs_override=None,
    n_initial=1000,
    batch_size=200,
    n_rounds=50,
    sample_k=60,
    n_workers=None,
    time_budget=60.0,
    seed=42,
):
    t0 = time.time()
    rng_np = np.random.RandomState(seed)
    round_history = []
    n_edges = len(eidx)
    edge_list = list(eidx)
    o = origin[0]

    if n_workers is None:
        n_workers = mp.cpu_count() if cfg.n_workers == -1 else cfg.n_workers

    # Build w capacity array.
    w_caps = [0.0] * n_edges
    for wkey, cap in w.items():
        edg, o_pass = wkey
        if o_pass != o:
            raise ValueError("w data should be filtered by origin before decomposing")
        w_caps[eidx[edg]] = cap

    # Per-bandwidth-constraint RHS for the master LP.
    if bc_rhs_override is not None:
        # Caller supplies the per-bandwidth-constraint budget directly (e.g. the residual
        # bandwidth left after another collective's flow has been subtracted off, for the
        # joint all-gather + all-to-all processing). Floored at 0 so a slightly-over-budget
        # constraint can't make the master infeasible.
        rhs = np.maximum(np.asarray(bc_rhs_override, dtype=float), 0.0)
    elif pooled_w_rhs:
        rhs = bc_coef.T @ np.asarray(w_caps)
    else:
        rhs = bc_bounds * R

    def duals_to_edge(master):
        # Convert per-bc duals back to a per-edge cost array so the existing per-edge tree
        # builders / reduced-cost test work unchanged: pi_edge[e] = sum_bc pi_bc * Acoef.
        return bc_coef @ np.asarray(master.get_duals_arr())

    T = max(node[1] for node in dests)
    dest_ipaths = {}
    for key, paths in f_paths.items():
        o_gpu, d_gpu = key
        if o_gpu == o:
            dest_ipaths[d_gpu, T] = paths

    # Destinations THIS origin actually reaches (its commodities), not the global union over all
    # origins.
    local_dests = list(dest_ipaths.keys())

    np_cost_bases = [
        np.ones(n_edges),
        np.array([1.0 / (w_caps[i] + 1e-12) for i in range(n_edges)]),
        np.zeros(n_edges),
    ]

    # Helper to generate cost arrays, selectors, orders for a batch
    def make_batch(count, pi_arr=None):
        costs = []
        sels = []
        orders = []
        for _ in range(count):
            if pi_arr is not None:
                noise = rng_np.exponential(0.3, size=n_edges)
                pi_floor = np.maximum(pi_arr, 0.01)
                ca = (pi_arr + noise * pi_floor).tolist()
                sel = "best" if rng_np.random() < 0.75 else "sample"
            else:
                base = np_cost_bases[rng_np.randint(0, 3)]
                ca = (base + rng_np.exponential(0.1, size=n_edges)).tolist()
                sel = rng_np.choice(["best", "sample", "random"])
            order = list(local_dests)
            rng_np.shuffle(order)
            costs.append(ca)
            sels.append(sel)
            orders.append(order)
        return costs, sels, orders

    def split_batch(costs, sels, orders, n_chunks, base_seed, pi_for_rc=None):
        """Split into n_chunks sub-batches for workers."""
        chunk_size = max(1, len(costs) // n_chunks)
        chunks = []
        for i in range(n_chunks):
            s = i * chunk_size
            e = s + chunk_size if i < n_chunks - 1 else len(costs)
            if s >= len(costs):
                break
            chunks.append(
                (costs[s:e], sels[s:e], orders[s:e], base_seed + i, sample_k, pi_for_rc)
            )
        return chunks

    def run_parallel(chunks, pool_proc):
        """Run worker batches, collect and deduplicate results."""
        all_results = pool_proc.map(_worker_build_batch, chunks)
        trees = []
        total_rc_rej = 0
        for results, n_rc_rej in all_results:
            total_rc_rej += n_rc_rej
            for (
                mult_list,
                dest_set,
                fp,
                dest_suffixes,
                parent,
                dest_full_edges,
            ) in results:
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    tree = _reconstruct_tree(
                        mult_list,
                        dest_set,
                        origin,
                        n_edges,
                        dest_suffixes,
                        parent,
                        dest_full_edges,
                    )
                    trees.append(tree)
        return trees, total_rc_rej

    # Phase 2: Initial set of trees
    logger.debug("Initial trees")
    t2 = time.time()

    pool = []
    seen_fps = set()

    costs, sels, orders = make_batch(n_initial)
    chunks = split_batch(costs, sels, orders, n_workers, seed)

    with mp.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(origin, local_dests, dest_ipaths, n_edges, eidx),
    ) as pool_proc:
        new_trees, _ = run_parallel(chunks, pool_proc)
        pool.extend(new_trees)

        t2d = time.time()

        if not pool:
            return [], 0.0, {"error": "no valid trees"}

        # Initial LP
        logger.debug("LP start")
        t3 = time.time()
        master = MasterLP(bc_coef, rhs, edge_list, gurobi_env)
        master.add_trees(pool)
        master.solve()
        obj = master.get_objective()
        pi_arr = duals_to_edge(master)
        pi = pi_arr  # per-edge duals (for the worker reduced-cost test)
        na = master.n_active()
        t3d = time.time()

        round_history.append(
            {"round": 0, "pool": len(pool), "obj": obj, "active": na, "time": t3d - t0}
        )

        # Column generation rounds
        stall = 0

        for rnd in range(1, n_rounds + 1):
            if time.time() - t0 > time_budget:
                logger.debug("  Time budget reached.")
                break
            tg = time.time()

            costs, sels, orders = make_batch(batch_size, pi_arr=pi_arr)
            chunks = split_batch(
                costs, sels, orders, n_workers, seed + rnd * 100, pi_for_rc=pi
            )

            new_trees, n_rc_rej = run_parallel(chunks, pool_proc)
            tgd = time.time()

            if not new_trees:
                stall += 1
                logger.debug(
                    "  R%d: no trees (%d/%d rc-rej), stall=%d, gen=%.1fs",
                    rnd,
                    n_rc_rej,
                    batch_size,
                    stall,
                    tgd - tg,
                )
                if stall >= 2:
                    # Exploration burst
                    burst_costs = [
                        rng_np.exponential(1.0, size=n_edges).tolist()
                        for _ in range(batch_size * 3)
                    ]
                    burst_sels = ["random"] * (batch_size * 3)
                    burst_orders = []
                    for _ in range(batch_size * 3):
                        order = list(local_dests)
                        rng_np.shuffle(order)
                        burst_orders.append(order)
                    bchunks = split_batch(
                        burst_costs,
                        burst_sels,
                        burst_orders,
                        n_workers,
                        seed + rnd * 1000,
                        pi_for_rc=pi,
                    )
                    new_trees, _ = run_parallel(bchunks, pool_proc)
                    if not new_trees:
                        logger.debug("  Burst empty. Stopping.")
                        break
                    stall = 0
                    logger.debug("  Burst found %d trees.", len(new_trees))
                else:
                    continue
            else:
                stall = 0

            tl = time.time()
            master.add_trees(new_trees)
            master.solve()
            new_obj = master.get_objective()
            pi_arr = duals_to_edge(master)
            pi = pi_arr
            na = master.n_active()
            tld = time.time()
            improvement = (new_obj - obj) / max(obj, 1e-12)
            obj = new_obj

            n_pruned = 0
            if na / len(master.trees) < 0.5:
                n_pruned = master.prune_inactive(rng_np, keep_fraction=0.1)
                if n_pruned > 0:
                    master.solve()
                    obj = master.get_objective()
                    pi_arr = duals_to_edge(master)
                    pi = pi_arr
                    na = master.n_active()

            round_history.append(
                {
                    "round": rnd,
                    "pool": len(master.trees),
                    "obj": obj,
                    "active": na,
                    "new": len(new_trees),
                    "time": time.time() - t0,
                }
            )

    solution = master.get_solution(edge_list)
    total_time = time.time() - t0

    return (
        solution,
        obj,
        dict(
            total_time=total_time,
            pool_size=len(master.trees),
            round_history=round_history,
            n_active=len(solution),
            objective=obj,
        ),
    )


def reverse_time_expanded(g, f, w, T):
    sdt = {(u, v): tv - tu for (u, tu), (v, tv) in g.edges()}
    g_rev = nx.DiGraph()
    g_rev.add_nodes_from(g.nodes())
    for (u, tu), (v, tv) in g.edges():
        g_rev.add_edge((v, T - tv), (u, T - tu))
    f_rev = {
        (j, i, T - (t + sdt[(i, j)]), d, o): val for (i, j, t, o, d), val in f.items()
    }
    w_rev = {(j, i, T - (t + sdt[(i, j)]), k): val for (i, j, t, k), val in w.items()}
    return g_rev, f_rev, w_rev


def _build_bc_coef(ec, bc_edge_sets, edge_list, n_bc):
    n_shifts = len(ec.gpus)
    spatial_coef = {}
    coef = np.zeros((len(edge_list), n_bc))
    for eid, ((u, _t), (v, _t2)) in enumerate(edge_list):
        se = (u, v)
        vec = spatial_coef.get(se)
        if vec is None:
            vec = np.zeros(n_bc)
            for i in range(n_shifts):
                sh = (ec.shift_fn(u, i), ec.shift_fn(v, i))
                for b, es in enumerate(bc_edge_sets):
                    if sh in es:
                        vec[b] += 1.0
            spatial_coef[se] = vec
        coef[eid] = vec
    return coef


def decompose_trees(
    top, g, f, w, collective, gurobi_env, cfg, R, full_budget=True, bc_rhs_override=None
):
    """Decompose the LP flow into WFQ trees.

    ``full_budget``: when True, the master LP packs trees against the full hardware budget.
    When False, or with multiple ECs, it instead uses each user's own orbit-summed ``w``
    allocation as the per-constraint budget.
    """
    logger.info("Begin decomposition")
    eidx = {e: i for i, e in enumerate(g.edges())}
    branching_nodes = determine_branching(top, g, collective)
    f_paths, dests = path_decomposition(g, f, branching_nodes, eidx, cfg)

    # compute some parameter values
    sample_k = top.n * 2
    n_rounds = top.n * 2
    time_limit = top.n * top.n / 4

    edge_list = list(eidx)
    bc_list = top.bandwidth_constraints
    bc_bounds = np.array([bc.bound for bc in bc_list], dtype=float)
    bc_edge_sets = [set(bc.edges) for bc in bc_list]
    pooled_w_rhs = (not full_budget) or (len(top.ECs) != 1)

    decomposed_trees = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        origin = (o, 0)
        w_pack = {}
        for ei, ej in g.edges():
            i, t = ei
            j, _ = ej
            w_pack[(ei, ej), o] = w.get((i, j, t, o), 0)
        bc_coef = _build_bc_coef(ec, bc_edge_sets, edge_list, len(bc_list))
        sol, obj, diag = pool_decompose(
            origin,
            dests,
            f_paths,
            w_pack,
            eidx,
            gurobi_env,
            cfg,
            bc_coef,
            bc_bounds,
            R,
            pooled_w_rhs,
            bc_rhs_override=bc_rhs_override,
            n_rounds=n_rounds,
            sample_k=sample_k,
            time_budget=time_limit,
        )
        decomposed_trees[o] = (sol, obj)
        logger.debug("decomposed origin %s objective %s", o, obj)

    return decomposed_trees
