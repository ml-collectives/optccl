from collections import Counter, defaultdict
from dataclasses import dataclass, field
import sys

# For small orbits we upgrade the transversal to a sharply-transitive (Latin)
# family of automorphisms.  A transversal alone is correct, but the group
# structure of a Latin set keeps the mirrored constraint matrix (A_sym/c_sym)
# well-conditioned, which matters enormously for the downstream Dantzig-Wolfe
# solver (an arbitrary transversal can make its master badly degenerate).  This
# is cheap for the small base topologies (and the production path builds ECs on
# a base, then propagates), while large direct detection keeps the fast
# transversal.
_LATIN_MAX_ORBIT = 16
_GROUP_SIZE_CAP = 50_000


@dataclass
class SymmetryProblem:
    """A labelled digraph plus the seed set whose orbits we want."""

    nodes: list
    node_type: dict
    edge_data: dict
    out_adj: dict
    in_adj: dict
    bc_index: dict
    bc_list: list
    seed_pool: list
    invariant_hooks: tuple = ()
    reject_hooks: tuple = ()
    verify_hooks: tuple = ()
    node_index: dict = field(default_factory=dict)


def build_problem(
    nodes,
    node_type,
    edge_data,
    bandwidth_constraints,
    seed_pool,
    *,
    invariant_hooks=(),
    reject_hooks=(),
    verify_hooks=(),
):
    """Precompute the immutable structures the search needs (done once)."""
    # Per-edge bandwidth-constraint signature: a sound label (a real
    # automorphism must map an edge to one with the same signature), verified
    # for real at the leaf via ``bc_index``.
    edge_bc_sigs = defaultdict(list)
    bc_index = {}
    bc_list = []
    for bc in bandwidth_constraints:
        sig = (bc.bound, len(bc.edges))
        for e in bc.edges:
            edge_bc_sigs[e].append(sig)
        bc_index[(frozenset(bc.edges), bc.bound)] = True
        bc_list.append((list(bc.edges), bc.bound))

    out_adj = {v: {} for v in nodes}
    in_adj = {v: {} for v in nodes}
    for (u, v), cost in edge_data.items():
        label = (cost, tuple(sorted(edge_bc_sigs.get((u, v), ()))))
        out_adj.setdefault(u, {})[v] = label
        in_adj.setdefault(v, {})[u] = label

    return SymmetryProblem(
        nodes=list(nodes),
        node_type=dict(node_type),
        edge_data=dict(edge_data),
        out_adj=out_adj,
        in_adj=in_adj,
        bc_index=bc_index,
        bc_list=bc_list,
        seed_pool=list(seed_pool),
        invariant_hooks=tuple(invariant_hooks),
        reject_hooks=tuple(reject_hooks),
        verify_hooks=tuple(verify_hooks),
        node_index={v: i for i, v in enumerate(nodes)},
    )


def refine_colors(p, individualize=(), intern=None):
    if intern is None:
        intern = {}

    def get_id(sig):
        i = intern.get(sig)
        if i is None:
            i = len(intern)
            intern[sig] = i
        return i

    indiv = {v: k + 1 for k, v in enumerate(individualize)}
    canon = {}
    for v in p.nodes:
        out_labels = tuple(sorted(p.out_adj.get(v, {}).values()))
        in_labels = tuple(sorted(p.in_adj.get(v, {}).values()))
        hook_vals = tuple(h(v) for h in p.invariant_hooks)
        canon[v] = get_id(
            ("i", indiv.get(v, 0), p.node_type[v], out_labels, in_labels, hook_vals)
        )
    n_classes = len(set(canon.values()))

    while True:
        nxt = {}
        for v in p.nodes:
            out_sig = tuple(
                sorted((el, canon[w]) for w, el in p.out_adj.get(v, {}).items())
            )
            in_sig = tuple(
                sorted((el, canon[w]) for w, el in p.in_adj.get(v, {}).items())
            )
            nxt[v] = get_id(("r", canon[v], out_sig, in_sig))
        new_n = len(set(nxt.values()))
        if new_n == n_classes:  # partition stopped getting finer -> stable
            return nxt
        canon, n_classes = nxt, new_n


def build_order(p, colors, seed):
    """Most-constrained-first ordering rooted at ``seed``.  Covers *all* nodes
    (including disconnected components) so produced maps are total."""
    class_size = Counter(colors.values())
    node_index = p.node_index
    assigned = {seed}
    order = [seed]
    remaining = [v for v in p.nodes if v != seed]

    def key(v):
        return (class_size[colors[v]], node_index[v])

    while remaining:
        frontier = [
            v
            for v in remaining
            if any(w in assigned for w in p.out_adj.get(v, {}))
            or any(w in assigned for w in p.in_adj.get(v, {}))
        ]
        nxt = min(frontier or remaining, key=key)
        order.append(nxt)
        assigned.add(nxt)
        remaining.remove(nxt)
    return order


def _consistent(p, nxt, label, assignment):
    lab_out = p.out_adj.get(label, {})
    for w, el in p.out_adj.get(nxt, {}).items():
        aw = assignment.get(w)
        if aw is not None and lab_out.get(aw) != el:
            return False
    lab_in = p.in_adj.get(label, {})
    for w, el in p.in_adj.get(nxt, {}).items():
        aw = assignment.get(w)
        if aw is not None and lab_in.get(aw) != el:
            return False
    return True


def verify_full(p, assignment):
    ed = p.edge_data
    for (u, v), cost in ed.items():
        mu = assignment.get(u)
        mv = assignment.get(v)
        if mu is None or mv is None or ed.get((mu, mv)) != cost:
            return False
    for edges, bound in p.bc_list:
        mapped = []
        for u, v in edges:
            mu = assignment.get(u)
            mv = assignment.get(v)
            if mu is None or mv is None:
                return False
            mapped.append((mu, mv))
        if (frozenset(mapped), bound) not in p.bc_index:
            return False
    for h in p.verify_hooks:
        if not h(assignment):
            return False
    return True


def find_automorphism(p, order, colors, color_class, fixed, extra_reject=()):
    """Iterative backtracking search for an automorphism extending ``fixed``."""
    assignment = dict(fixed)
    used = set(assignment.values())
    so = [v for v in order if v not in assignment]
    n = len(so)
    rejects = p.reject_hooks + tuple(extra_reject)
    if n == 0:
        return dict(assignment) if verify_full(p, assignment) else None

    def candidates(i):
        nxt = so[i]
        res = []
        for label in color_class.get(colors[nxt], ()):
            if label in used:
                continue
            if _consistent(p, nxt, label, assignment) and not any(
                r(nxt, label, assignment) for r in rejects
            ):
                res.append(label)
        return res

    cand = [None] * n
    idx = [0] * n
    cand[0] = candidates(0)
    i = 0
    while i >= 0:
        if idx[i] < len(cand[i]):
            label = cand[i][idx[i]]
            idx[i] += 1
            if label in used:  # freed/taken since candidates() was computed
                continue
            nxt = so[i]
            assignment[nxt] = label
            used.add(label)
            if i == n - 1:
                if verify_full(p, assignment):
                    return dict(assignment)
                del assignment[nxt]
                used.discard(label)
                continue
            i += 1
            cand[i] = candidates(i)
            idx[i] = 0
        else:
            i -= 1
            if i >= 0:
                nxt = so[i]
                used.discard(assignment.pop(nxt))
    return None


def _identity_map(p):
    return {v: v for v in p.nodes}


def _hash_map(d):
    return tuple(sorted(d.items(), key=lambda kv: repr(kv[0])))


def _compose(s, g):
    # (s after g): node -> s[g[node]]
    return {k: s[v] for k, v in g.items()}


def _group_closure(generators, cap):
    """Close ``generators`` (full-node maps) under composition; the identity is
    element 0.  Returns ``None`` if the group would exceed ``cap`` elements."""
    gens = list(generators)
    if not gens:
        return None
    ident = {k: k for k in gens[0]}
    seen = {_hash_map(ident)}
    group = [ident]
    frontier = [ident]
    while frontier:
        new = []
        for g in frontier:
            for s in gens:
                x = _compose(s, g)
                h = _hash_map(x)
                if h not in seen:
                    seen.add(h)
                    group.append(x)
                    new.append(x)
                    if len(group) > cap:
                        return None
        frontier = new
    return group


def _latin_square(group, members):
    """Pick one map per member so the chosen maps are sharply transitive on
    ``members`` (a Latin square).  ``group[0]`` is the identity, so row 0 is the
    identity.  Returns the list of maps, or ``None`` if no such set exists."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, len(members) + 100))
    try:

        def recurse(available, square):
            if len(square) == len(members):
                return square
            for l in available:
                if any(l[el] == sq[el] for el in members for sq in square):
                    continue
                res = recurse([x for x in available if x is not l], square + [l])
                if res is not None:
                    return res
            return None

        return recurse(list(group), [])
    finally:
        sys.setrecursionlimit(limit)


def _try_latin(members, transversal):
    """Upgrade a small orbit's transversal to a sharply-transitive Latin family
    for better solver conditioning, or ``None`` to keep the plain transversal."""
    if len(members) > _LATIN_MAX_ORBIT or not transversal:
        return None
    group = _group_closure(list(transversal.values()), _GROUP_SIZE_CAP)
    if group is None:
        return None
    return _latin_square(group, members)


def compute_equivalence_classes(p):
    intern = {}
    global_colors = refine_colors(p, intern=intern)  # cheap orbit prefilter

    remaining = list(p.seed_pool)
    remaining_set = set(remaining)
    result = []
    while remaining:
        seed = remaining[0]
        # Domain colouring with the representative pinned: sharpens ordering and
        # candidate images for every search rooted at this seed.
        colors_dom = refine_colors(p, individualize=(seed,), intern=intern)
        order = build_order(p, colors_dom, seed)

        members = [seed]
        rows = [_identity_map(p)]
        transversal = {}
        for tgt in remaining:
            if tgt == seed or global_colors[tgt] != global_colors[seed]:
                continue
            colors_cod = refine_colors(p, individualize=(tgt,), intern=intern)
            cod_class = defaultdict(list)
            for v in p.nodes:
                cod_class[colors_cod[v]].append(v)
            auto = find_automorphism(p, order, colors_dom, cod_class, {seed: tgt})
            if auto is not None:  # tgt is genuinely in the orbit
                members.append(tgt)
                rows.append(auto)
                transversal[tgt] = auto

        # Small orbits: upgrade to a sharply-transitive Latin family (better DW
        # conditioning); large orbits keep the fast transversal.
        latin = _try_latin(members, transversal)
        if latin is not None:
            rows = latin

        result.append((members, rows))
        for g in members:
            remaining_set.discard(g)
        remaining = [g for g in remaining if g in remaining_set]
    return result
