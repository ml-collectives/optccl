from .. import symmetry
from ..topologies import EC

_DEMAND_HEAD = "__demand__"


def build_workload_symmetry_problem(wl) -> symmetry.SymmetryProblem:
    top = wl.top
    nodes = list(top.components)
    node_type = {c: ("comp",) + tuple(sorted(top.capabilities[c])) for c in nodes}
    edge_data = {e: ("e", c) for e, c in top.edge_data.items()}

    for k, ce in enumerate(wl.collectives):
        dn = (_DEMAND_HEAD, k)
        assert dn not in node_type, f"component id collides with demand node {dn}"
        nodes.append(dn)
        node_type[dn] = ("demand", ce.kind, ce.demand)
        roles = {}
        if ce.participants is not None:
            for p in ce.participants:
                roles.setdefault(p, []).append("part")
        else:
            roles.setdefault(ce.root, []).append("root")
            for p in ce.leaves:
                roles.setdefault(p, []).append("leaf")
        for p, rr in roles.items():
            edge_data[(dn, p)] = ("d", ce.kind, tuple(sorted(rr)), ce.demand)

    return symmetry.build_problem(
        nodes,
        node_type,
        edge_data,
        top.bandwidth_constraints,
        seed_pool=sorted(wl.effective_gpus, key=repr),
    )


def generate_workload_ecs(wl) -> list:
    p = build_workload_symmetry_problem(wl)
    ecs = []
    for members, rows in symmetry.compute_equivalence_classes(p):
        # default-arg binding captures this class's members/rows (not the last
        # loop iteration's) -- the shift map is a lookup into rows.
        def fn(node, shift, members=members, rows=rows):
            return rows[shift % len(members)][node]

        ecs.append(EC(members, fn))
    return ecs


def validate_workload_ecs(wl):
    assert wl.ECs is not None, "workload has no ECs"
    p = build_workload_symmetry_problem(wl)
    all_nodes = set(p.nodes)

    members_all = [g for ec in wl.ECs for g in ec.gpus]
    assert set(members_all) == set(wl.effective_gpus), (
        "ECs do not cover the effective GPUs"
    )
    assert len(members_all) == len(set(members_all)), "ECs overlap"

    for ec in wl.ECs:
        m = len(ec.gpus)
        for shift in range(m):
            perm = {v: ec.shift_fn(v, shift) for v in p.nodes}
            assert set(perm.values()) == all_nodes, (
                "shift map is not a bijection over all nodes"
            )
            if shift == 0:
                assert all(perm[v] == v for v in p.nodes), "shift 0 is not the identity"
            assert symmetry.verify_full(p, perm), (
                "shift map is not a demand-preserving topology automorphism"
            )
        rep = ec.gpus[0]
        images = {ec.shift_fn(rep, k) for k in range(m)}
        assert images == set(ec.gpus), "shifts do not form a transversal of the class"
