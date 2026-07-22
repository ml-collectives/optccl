import logging
from collections import defaultdict

import gurobipy as gp

from .topologies import EC, BandwidthConstraint, Topology

logger = logging.getLogger(__name__)


def _dup_vc(v, u):
    return ("dup", v, ("vc", u))


def _dup_vcr(v, u):
    return ("dup", v, ("vcr", u))


def _dup_dst(x, group):
    return ("dup", x, ("dst", tuple(sorted(group, key=repr))))


def _dup_in(x, t, p):
    return ("dup", x, ("in", t, p))


def _dup_out(x, t, q):
    return ("dup", x, ("out", t, q))


def _is_dup(node):
    return isinstance(node, tuple) and len(node) == 3 and node[0] == "dup"


def original_component(node):
    return node[1] if _is_dup(node) else node


class StrengthenState:
    """Cumulative duplication decisions, keyed by ORIGINAL components.

    The strengthened topology is always rebuilt from the base topology plus
    this state, so duplicate ids stay one level deep and offending vertices
    re-detected on a strengthened topology fold back into the same entries.
    """

    def __init__(self):
        # non-copy components replaced by vc-tagged duplicates (fan-out).
        self.vc_duplicated = set()
        # non-reduce components replaced by vcr-tagged duplicates (fan-in).
        self.vcr_duplicated = set()
        # (copy-capable storage component, t) split by incoming instance.
        self.in_splits = set()
        # (reduce-capable storage component, t) split by outgoing instance.
        self.out_splits = set()
        # copy-capable component -> list[frozenset] partitioning all GPUs
        # (dst scheme; dormant).
        self.dest_partitions = {}

    def snapshot(self):
        return (
            frozenset(self.vc_duplicated),
            frozenset(self.vcr_duplicated),
            frozenset(self.in_splits),
            frozenset(self.out_splits),
            {x: frozenset(p) for x, p in self.dest_partitions.items()},
        )

    def describe(self):
        parts = []
        if self.vc_duplicated:
            parts.append(f"{len(self.vc_duplicated)} non-copy vertex(es) duplicated")
        if self.vcr_duplicated:
            parts.append(f"{len(self.vcr_duplicated)} non-reduce vertex(es) duplicated")
        if self.in_splits:
            parts.append(f"{len(self.in_splits)} (vertex, t) split by in-neighbor")
        if self.out_splits:
            parts.append(f"{len(self.out_splits)} (vertex, t) split by out-neighbor")
        if self.dest_partitions:
            parts.append(f"{len(self.dest_partitions)} vertex(es) destination-split")
        return ", ".join(parts) if parts else "no duplications"


def _split_partition(partition, cut):
    out = []
    changed = False
    for grp in partition:
        a = grp & cut
        b = grp - cut
        if a and b:
            out.append(a)
            out.append(b)
            changed = True
        else:
            out.append(grp)
    return out, changed


def _shift_perms(base_top):
    perms = []
    for ec in base_top.ECs:
        for s in range(1, len(ec.gpus)):
            fwd = {n: ec.shift_fn(n, s) for n in base_top.components}
            perms.append(fwd)
            perms.append({v: k for k, v in fwd.items()})
    return perms


def _close_under_symmetry(state, base_top):
    """Make the state equivariant: duplicate whole orbits, with tag/partition
    images matching exactly. Terminates -- the duplication sets only grow and
    partitions only refine."""
    perms = _shift_perms(base_top)
    all_gpus = frozenset(base_top.gpus)
    changed = True
    while changed:
        changed = False
        for perm in perms:
            for vset in (state.vc_duplicated, state.vcr_duplicated):
                for v in list(vset):
                    iv = perm[v]
                    if iv not in vset:
                        vset.add(iv)
                        changed = True
            for sset in (state.in_splits, state.out_splits):
                for x, t in list(sset):
                    key = (perm[x], t)
                    if key not in sset:
                        sset.add(key)
                        changed = True
            for x in list(state.dest_partitions):
                part = state.dest_partitions[x]
                ix = perm[x]
                img = [frozenset(perm[d] for d in grp) for grp in part]
                cur = state.dest_partitions.get(ix, [all_gpus])
                any_split = False
                for grp in img:
                    cur, c = _split_partition(cur, grp)
                    any_split = any_split or c
                if any_split:
                    state.dest_partitions[ix] = cur
                    changed = True
    for x in list(state.dest_partitions):
        if len(state.dest_partitions[x]) <= 1:
            del state.dest_partitions[x]


def _out_adjacency(base_top):
    out_adj = defaultdict(list)
    for a, b in base_top.edge_data:
        if a != b:
            out_adj[a].append(b)
    return out_adj


def _in_adjacency(base_top):
    in_adj = defaultdict(list)
    for a, b in base_top.edge_data:
        if a != b:
            in_adj[b].append(a)
    return in_adj


def _in_neighbors(base_top):
    in_adj = defaultdict(set)
    for a, b in base_top.edge_data:
        in_adj[b].add(a)
    return in_adj


def _out_neighbors(base_top):
    out_adj = defaultdict(set)
    for a, b in base_top.edge_data:
        out_adj[a].add(b)
    return out_adj


def _vc_tags(v, dup_set, adj, target_set):
    tags = set()
    seen = {v}
    stack = [v]
    while stack:
        cur = stack.pop()
        for b in adj.get(cur, ()):
            if b in target_set:
                tags.add(b)
            elif b in dup_set and b not in seen:
                seen.add(b)
                stack.append(b)
    return tags


def _resolve_conflicts(state, base_top):
    both_switch = state.vc_duplicated & state.vcr_duplicated
    if both_switch:
        logger.warning(
            "strengthening: %d vertex(es) proposed for both vc and vcr "
            "duplication; keeping vc only: %s",
            len(both_switch),
            sorted(both_switch, key=repr),
        )
        state.vcr_duplicated -= both_switch
    both_split = state.in_splits & state.out_splits
    if both_split:
        logger.warning(
            "strengthening: %d (vertex, t) pair(s) proposed for both in- and "
            "out-splitting; keeping the in-split only: %s",
            len(both_split),
            sorted(both_split, key=repr),
        )
        state.out_splits -= both_split
    spatial = state.vc_duplicated | state.vcr_duplicated | set(state.dest_partitions)
    for sset, kind in ((state.in_splits, "in"), (state.out_splits, "out")):
        clash = {(x, t) for (x, t) in sset if x in spatial}
        if clash:
            logger.warning(
                "strengthening: dropping %d %s-split(s) on vertices that are "
                "already spatially duplicated: %s",
                len(clash),
                kind,
                sorted(clash, key=repr),
            )
            sset -= clash


def _prune_untaggable(state, base_top):
    out_adj = _out_adjacency(base_top)
    in_adj = _in_adjacency(base_top)
    for dup_set, adj, targets, kind in (
        (state.vc_duplicated, out_adj, base_top.copy_nodes, "vc"),
        (state.vcr_duplicated, in_adj, base_top.reduce_nodes, "vcr"),
    ):
        while True:
            empty = [v for v in dup_set if not _vc_tags(v, dup_set, adj, targets)]
            if not empty:
                break
            for v in empty:
                logger.warning(
                    "strengthening: not duplicating %r (%s) -- no tag-capable "
                    "vertex is reachable, so duplication cannot separate its "
                    "flows",
                    v,
                    kind,
                )
                dup_set.discard(v)
    in_nbrs = _in_neighbors(base_top)
    out_nbrs = _out_neighbors(base_top)
    for sset, nbrs, kind in (
        (state.in_splits, in_nbrs, "in"),
        (state.out_splits, out_nbrs, "out"),
    ):
        for x, t in list(sset):
            if not nbrs.get(x):
                logger.warning(
                    "strengthening: not %s-splitting %r at t=%d -- it has no "
                    "edges to partition",
                    kind,
                    x,
                    t,
                )
                sset.discard((x, t))


def refine_state(state, proposals, base_top):
    """Fold detection proposals into the state and re-establish its
    invariants (symmetry closure, conflict resolution, no futile
    duplications). Returns whether the state actually changed; False means
    strengthening cannot make progress."""
    before = state.snapshot()
    all_gpus = frozenset(base_top.gpus)
    for prop in proposals:
        if prop[0] == "vc":
            state.vc_duplicated.add(prop[1])
        elif prop[0] == "vcr":
            state.vcr_duplicated.add(prop[1])
        elif prop[0] == "in":
            state.in_splits.add((prop[1], prop[2]))
        elif prop[0] == "out":
            state.out_splits.add((prop[1], prop[2]))
        else:
            _, x, groups = prop
            part = state.dest_partitions.get(x, [all_gpus])
            for grp in groups:
                part, _ = _split_partition(part, grp)
            if len(part) > 1:
                state.dest_partitions[x] = part
    _close_under_symmetry(state, base_top)
    _resolve_conflicts(state, base_top)
    _prune_untaggable(state, base_top)
    return state.snapshot() != before


def apply_strengthening(base_top: Topology, state: StrengthenState, T) -> Topology:
    """Rebuild the topology from ``base_top`` with the state's duplications
    applied, propagating (wrapping) the base ECs. ``T`` is the time horizon
    the topology will be constructed with (per-layer splits are expressed
    through ``edge_layers`` relative to it). The caller should assert
    ``validate_ecs`` on the result."""
    out_adj = _out_adjacency(base_top)
    in_adj = _in_adjacency(base_top)
    copy_nodes = base_top.copy_nodes
    reduce_nodes = base_top.reduce_nodes
    storage = base_top.storage_nodes

    vc_tags = {
        v: sorted(_vc_tags(v, state.vc_duplicated, out_adj, copy_nodes), key=repr)
        for v in state.vc_duplicated
    }
    vcr_tags = {
        v: sorted(_vc_tags(v, state.vcr_duplicated, in_adj, reduce_nodes), key=repr)
        for v in state.vcr_duplicated
    }
    parts = {x: sorted(p, key=repr) for x, p in state.dest_partitions.items()}
    in_split_at = defaultdict(set)
    for x, t in state.in_splits:
        in_split_at[x].add(t)
    out_split_at = defaultdict(set)
    for x, t in state.out_splits:
        out_split_at[x].add(t)
    in_tags = {x: sorted(_in_neighbors(base_top)[x], key=repr) for x in in_split_at}
    out_tags = {x: sorted(_out_neighbors(base_top)[x], key=repr) for x in out_split_at}

    components = []
    capabilities = {}
    dest_filter = {}
    for c in base_top.components:
        if c in vc_tags or c in vcr_tags:
            tags = vc_tags.get(c, ())
            for u in tags:
                node = _dup_vc(c, u)
                components.append(node)
                capabilities[node] = set(base_top.capabilities[c])
            for u in vcr_tags.get(c, ()):
                node = _dup_vcr(c, u)
                components.append(node)
                capabilities[node] = set(base_top.capabilities[c])
            continue
        components.append(c)
        capabilities[c] = set(base_top.capabilities[c])
        if c in parts:
            for grp in parts[c]:
                node = _dup_dst(c, grp)
                components.append(node)
                capabilities[node] = {"copy"}
                dest_filter[node] = frozenset(grp)
        for t in sorted(in_split_at.get(c, ())):
            for p in in_tags[c]:
                node = _dup_in(c, t, p)
                components.append(node)
                capabilities[node] = set(base_top.capabilities[c]) - {"gpu"}
        for t in sorted(out_split_at.get(c, ())):
            for q in out_tags[c]:
                node = _dup_out(c, t, q)
                components.append(node)
                capabilities[node] = set(base_top.capabilities[c]) - {"gpu"}

    comp_set = set(components)
    # A duplicated nic can no longer be referenced by name
    nics = [n for n in base_top.nics if n in comp_set]

    def tails_at(a, t):
        if t in in_split_at.get(a, ()):
            return [_dup_in(a, t, p) for p in in_tags[a]]
        if t in out_split_at.get(a, ()):
            return [_dup_out(a, t, q) for q in out_tags[a]]
        if a in vc_tags or a in vcr_tags:
            return [_dup_vc(a, u) for u in vc_tags.get(a, ())] + [
                _dup_vcr(a, u) for u in vcr_tags.get(a, ())
            ]
        if a in parts:
            return [_dup_dst(a, grp) for grp in parts[a]]
        return [a]

    def heads_at(b, tb, a):
        # In-split heads take the single copy tagged by the SPATIAL tail, so
        # e.g. all persistence loops of layer t's copies merge into the one
        # self-tagged copy (or plain x) of layer t+1. Out-split heads take
        # every copy: in-edges enter all of them (usage splits fractionally,
        # so a single arrival still pays once).
        if tb in in_split_at.get(b, ()):
            return [_dup_in(b, tb, a)]
        if tb in out_split_at.get(b, ()):
            return [_dup_out(b, tb, q) for q in out_tags[b]]
        if b in vc_tags or b in vcr_tags:
            return [_dup_vc(b, u) for u in vc_tags.get(b, ())] + [
                _dup_vcr(b, u) for u in vcr_tags.get(b, ())
            ]
        return [b]

    def edge_allowed(ta, hb):
        # vc-tagged tails may only lead toward their tag (matched through
        # other vc duplicates; dropped into non-duplicated non-copy vertices).
        if _is_dup(ta) and ta[2][0] == "vc":
            tag = ta[2][1]
            if _is_dup(hb):
                if hb[2][0] == "vc":
                    return hb[2] == ("vc", tag)
                # per-layer copy of a copy-capable vertex: match spatially.
                if original_component(hb) in copy_nodes:
                    return original_component(hb) == tag
                return True
            if hb in copy_nodes:
                return hb == tag
            return True
        # out-tagged tails own exactly their tagged neighbor's edge.
        if _is_dup(ta) and ta[2][0] == "out":
            if original_component(hb) != ta[2][2]:
                return False
            # fall through: the head may impose vcr rules too.
        # vcr-tagged heads may only be fed from their tag side (mirror).
        if _is_dup(hb) and hb[2][0] == "vcr":
            tag = hb[2][1]
            if _is_dup(ta) and ta[2][0] == "vcr":
                return ta[2] == ("vcr", tag)
            sa = original_component(ta)
            if sa in reduce_nodes:
                return sa == tag
            return True
        return True

    inst_layers = defaultdict(set)
    inst_cost = {}
    emap = defaultdict(set)
    for (a, b), cost in base_top.edge_data.items():
        for t in range(T):
            tb = t + 1 if b in storage else t
            if a == b and a in parts:
                # Persistence self-loops of dst-split vertices stay on the hub.
                pairs = [(a, b)]
            else:
                pairs = [
                    (ta, hb)
                    for ta in tails_at(a, t)
                    for hb in heads_at(b, tb, a)
                    if edge_allowed(ta, hb)
                ]
            for pr in pairs:
                inst_layers[pr].add(t)
                inst_cost[pr] = cost
                emap[(a, b)].add(pr)
    for x in parts:
        for grp in parts[x]:
            pr = (x, _dup_dst(x, grp))
            inst_layers[pr].update(range(T))
            inst_cost[pr] = 0

    edge_data = {pr: inst_cost[pr] for pr in inst_layers}
    edge_layers = {
        pr: frozenset(layers) for pr, layers in inst_layers.items() if len(layers) < T
    }

    # Every duplicated instance shares the original edge's bandwidth
    # constraints -- capacity is split among the copies, never multiplied.
    # Each constraint additionally lists the FLIPS of the opposite
    # direction's instances.
    bcs = []
    for bc in base_top.bandwidth_constraints:
        edges = []
        seen_e = set()
        for e in bc.edges:
            for pr in sorted(emap[e], key=repr):
                if pr not in seen_e:
                    seen_e.add(pr)
                    edges.append(pr)
            a, b = e
            for A, B in sorted(emap.get((b, a), ()), key=repr):
                pr = (B, A)
                if pr not in seen_e:
                    seen_e.add(pr)
                    edges.append(pr)
        bcs.append(BandwidthConstraint(bc.name, edges, bc.bound))

    new_top = Topology(components, capabilities, nics, edge_data, bcs)
    new_top.dest_filter = dest_filter
    new_top.edge_layers = edge_layers
    new_top.add_ECs(_wrap_ecs(base_top))
    return new_top


def _wrap_ecs(base_top):
    """Propagate the base ECs onto a strengthened topology: shifts act on a
    duplicate by shifting its base component and its tag (timesteps are held
    fixed, as everywhere in the EC machinery). Well-defined because the state
    is closed under the shift group (see _close_under_symmetry). Handles
    every duplicate kind, so it also covers topologies merged from several
    strengthened phases."""
    new_ecs = []
    for ec in base_top.ECs:
        base_fn = ec.shift_fn

        def fn(node, shift, base_fn=base_fn):
            if _is_dup(node):
                _, x, tag = node
                sx = base_fn(x, shift)
                if tag[0] in ("vc", "vcr"):
                    return ("dup", sx, (tag[0], base_fn(tag[1], shift)))
                if tag[0] in ("in", "out"):
                    return ("dup", sx, (tag[0], tag[1], base_fn(tag[2], shift)))
                return (
                    "dup",
                    sx,
                    (
                        "dst",
                        tuple(sorted((base_fn(d, shift) for d in tag[1]), key=repr)),
                    ),
                )
            return base_fn(node, shift)

        new_ecs.append(EC(list(ec.gpus), fn))
    return new_ecs


def merge_strengthened_topologies(base_top: Topology, tops) -> Topology:
    """Union of independently strengthened variants of one base topology, for
    processing phases that were solved on their own topologies (the separate
    all_reduce LPs)."""
    components = []
    capabilities = {}
    seen = set()
    for top in tops:
        for c in top.components:
            if c not in seen:
                seen.add(c)
                components.append(c)
                capabilities[c] = set(top.capabilities[c])
    nics = [n for n in base_top.nics if n in seen]

    edge_data = {}
    edge_layers = {}
    unrestricted = set()
    for top in tops:
        for e, cost in top.edge_data.items():
            edge_data[e] = cost
            if e in unrestricted:
                continue
            layers = top.edge_layers.get(e)
            if layers is None:
                unrestricted.add(e)
                edge_layers.pop(e, None)
            else:
                edge_layers[e] = edge_layers.get(e, frozenset()) | layers

    bcs = []
    for i, bc in enumerate(base_top.bandwidth_constraints):
        edges = []
        eseen = set()
        for top in tops:
            for e in top.bandwidth_constraints[i].edges:
                if e not in eseen:
                    eseen.add(e)
                    edges.append(e)
        bcs.append(BandwidthConstraint(bc.name, edges, bc.bound))

    merged = Topology(components, capabilities, nics, edge_data, bcs)
    dest_filter = {}
    for top in tops:
        dest_filter.update(top.dest_filter)
    merged.dest_filter = dest_filter
    merged.edge_layers = edge_layers
    merged.add_ECs(_wrap_ecs(base_top))
    return merged


def check_full_volume(decomposed, tol):
    return {o: obj for o, (_sol, obj) in decomposed.items() if obj < 1 - tol}


def _tight_edges(sol, w_edge, gurobi_env, tol):
    m = gp.Model("strengthen_diag", env=gurobi_env)
    m.setParam("OutputFlag", 0)
    m.ModelSense = gp.GRB.MAXIMIZE
    lam = [m.addVar(lb=0.0, obj=1.0) for _ in sol]
    per_edge = defaultdict(list)
    for i, (edge_dict, _vol, _sfx, _ecd) in enumerate(sol):
        for e, mult in edge_dict.items():
            per_edge[e].append((i, mult))
    constrs = {}
    for e, terms in per_edge.items():
        constrs[e] = m.addConstr(
            gp.quicksum(lam[i] * mult for i, mult in terms) <= w_edge.get(e, 0)
        )
    m.optimize()
    if m.status != gp.GRB.OPTIMAL:
        return []
    return [e for e, c in constrs.items() if abs(c.Pi) > tol]


def _classify(e, base_top, orientation, T):
    (x, t), (y, _t2) = e
    ox = original_component(x)
    oy = original_component(y)
    if ox == oy:
        # Persistence self-loop or internal hub edge: bandwidth-free, nothing
        # to separate.
        return None
    merge_nodes = (
        base_top.copy_nodes if orientation == "fanout" else base_top.reduce_nodes
    )
    if oy not in merge_nodes:
        if _is_dup(y):
            logger.warning(
                "strengthening: offending edge into %r, which is already a "
                "duplicate -- no further refinement available there",
                y,
            )
            return None
        return ("vc" if orientation == "fanout" else "vcr", oy)
    # Merge-capable tail: split the offending layer by transmission instance.
    if t <= 0:
        # Layer 0 of the decomposition graph only holds the (single) injected
        # instance -- nothing to separate.
        logger.debug("strengthening: offending edge at t=0 out of %r skipped", x)
        return None
    if ox not in base_top.storage_nodes:
        if ox not in merge_nodes and not _is_dup(x):
            # Non-storage pass-through tail feeding a merge-capable head: the
            # switch scheme is the one that can separate its flows.
            return ("vc" if orientation == "fanout" else "vcr", ox)
        logger.warning(
            "strengthening: offending tail %r has no storage; per-layer "
            "splitting does not apply",
            ox,
        )
        return None
    if orientation == "fanout":
        if _is_dup(x) and x[2][0] == "in" and x[2][1] == t:
            logger.warning(
                "strengthening: %r is already split by in-neighbor at t=%d -- "
                "no further refinement available there",
                ox,
                t,
            )
            return None
        return ("in", ox, t)
    t_fwd = T - t
    if _is_dup(x) and x[2][0] == "out" and x[2][1] == t_fwd:
        logger.warning(
            "strengthening: %r is already split by out-neighbor at t=%d -- "
            "no further refinement available there",
            ox,
            t_fwd,
        )
        return None
    return ("out", ox, t_fwd)


def detect_offending(decomposed, w, g, base_top, cfg, gurobi_env, orientation="fanout"):
    """Find and classify the offending edges of a shortfall decomposition."""
    tol = cfg.strengthen_tol
    T = max(t for _n, t in g.nodes())
    proposals = []
    for o, (sol, obj) in decomposed.items():
        if obj >= 1 - tol:
            continue
        w_edge = {}
        for ei, ej in g.edges():
            i, t = ei
            j, _ = ej
            w_edge[(ei, ej)] = w.get((i, j, t, o), 0)
        load = defaultdict(float)
        for edge_dict, vol, _sfx, _ecd in sol:
            for e, mult in edge_dict.items():
                load[e] += vol * mult
        # The packed trees carry only obj (<1) volume; scale to the full unit
        # the schedule will actually push (post_process rescales by 1/obj).
        scale = 1.0 / max(obj, tol)
        offending = [e for e, ld in load.items() if ld * scale > w_edge.get(e, 0) + tol]
        if not offending:
            offending = _tight_edges(sol, w_edge, gurobi_env, tol)
            if offending:
                logger.info(
                    "strengthening: origin %r: no edge load exceeds its w; "
                    "using %d dual-tight edge(s) from the w-capacitated repack",
                    o,
                    len(offending),
                )
        if not offending:
            logger.warning(
                "strengthening: origin %r recovered only %.4f volume but no "
                "offending edge was found -- the shortfall does not look like "
                "a w-merging artifact, duplication will not help",
                o,
                obj,
            )
            continue
        for e in offending:
            prop = _classify(e, base_top, orientation, T)
            if prop is not None:
                proposals.append(prop)
    return proposals
