import networkx as nx

from collections import Counter

from . import symmetry
from .errors import OptcclError


class BandwidthConstraint:
    def __init__(self, name: str, edges, bound):
        self.name = name
        self.edges = edges
        self.bound = bound


class EC:
    def __init__(self, gpus, fn):
        self.gpus = gpus
        self.shift_fn = fn


def ec_edge_preimage_maps(ECs, edges):
    """Per EC representative, one inverse edge map per shift:
    ``maps[rep][s][e]`` is the representative's edge whose image under shift
    ``s`` is the physical edge ``e``."""
    edges = list(edges)
    maps = {}
    for ec in ECs:
        maps[ec.gpus[0]] = [
            {(ec.shift_fn(a, s), ec.shift_fn(b, s)): (a, b) for a, b in edges}
            for s in range(len(ec.gpus))
        ]
    return maps


CAPABILITIES = frozenset({"gpu", "storage", "copy", "reduce"})


def _normalize_capabilities(caps):
    caps = set(caps)
    unknown = caps - CAPABILITIES
    if unknown:
        raise OptcclError(f"Unknown capabilities: {sorted(unknown)}")
    if "gpu" in caps:
        caps |= {"storage", "copy", "reduce"}
    return caps


class Topology:
    def __init__(
        self,
        components: list,
        capabilities: dict,
        nics: list,
        edge_data,
        bandwidth_constraints,
    ):
        """
        Constructor for Topology class. Used to describe the underlying
        structure of the topology, what constraints exist, and what our
        objective is.

        A topology is a flat list of *components*, each declaring a set of
        capabilities drawn from :data:`CAPABILITIES`:

          - ``gpu``     -- a source/sink of collectives (a demand endpoint).
          - ``storage`` -- occupies a layer in the time-expanded network, i.e.
                            data can persist across a timestep here.
          - ``copy``    -- may fan-out / duplicate data (AllGather).
          - ``reduce``  -- may fan-in / combine data (ReduceScatter/AllReduce).

        Declaring ``gpu`` implies all four capabilities. A component with no
        capabilities is a pure pass-through switch.

        Arguments:
        components -- list of node ids
        capabilities -- dict mapping each component to its set of capabilities
        nics -- subset of components used to connect to a multirail topology
        edge_data -- dictionary that maps from edges to their objective cost
        bandwidth_constraints -- list of BandwidthConstraint objects
        """

        self.components = list(components)
        self.capabilities = {
            c: _normalize_capabilities(capabilities.get(c, ())) for c in self.components
        }

        comp_set = set(self.components)
        if len(comp_set) != len(self.components):
            dupes = sorted(
                (c for c, n in Counter(self.components).items() if n > 1), key=repr
            )
            raise OptcclError(f"duplicate component id(s): {dupes}")
        undeclared = set(nics) - comp_set
        if undeclared:
            raise OptcclError(
                f"nic(s) not declared as components: {sorted(undeclared, key=repr)}"
            )
        self.nics = nics

        # Capability-derived node sets consumed by the LP builders.
        self.gpus = [c for c in self.components if "gpu" in self.capabilities[c]]
        self.storage_nodes = set(
            c for c in self.components if "storage" in self.capabilities[c]
        )
        self.copy_nodes = set(
            c for c in self.components if "copy" in self.capabilities[c]
        )
        self.reduce_nodes = set(
            c for c in self.components if "reduce" in self.capabilities[c]
        )

        self.ECs = None
        self.n = len(self.gpus)

        # component -> frozenset of allowed destination GPUs.
        self.dest_filter = {}

        # edge -> frozenset of tail timesteps at which the edge exists in the
        # time-expanded network.
        self.edge_layers = {}

        # An edge advances a timestep iff its head has storage. The nn/ns/sn/ss
        # buckets encode that: "n" = storage nodes, "s" = non-storage (switch).
        self.nnpairs = []
        self.nspairs = []
        self.snpairs = []
        self.sspairs = []

        self.edge_data = edge_data

        storage = self.storage_nodes
        for edge in edge_data.keys():
            u, v = edge
            if u not in comp_set or v not in comp_set:
                raise OptcclError(f"Node not declared: {u} or {v}")
            if u in storage and v in storage:
                self.nnpairs.append(edge)
            elif u in storage and v not in storage:
                self.nspairs.append(edge)
            elif u not in storage and v in storage:
                self.snpairs.append(edge)
            else:
                self.sspairs.append(edge)

        # define bandwidth groupings and corresponding bandwidths
        self.bandwidth_constraints = bandwidth_constraints
        for bc in self.bandwidth_constraints:
            for edge in bc.edges:
                if edge not in edge_data and (edge[1], edge[0]) not in edge_data:
                    raise OptcclError(
                        f"bandwidth constraint {bc.name!r} references edge "
                        f"{edge!r}, which is not in the topology's edge set"
                    )

    @classmethod
    def from_node_types(
        cls, gpus, mems, switches, nics, edge_data, bandwidth_constraints
    ):
        assert set(gpus).isdisjoint(set(mems))
        assert set(gpus).isdisjoint(set(switches))
        assert set(mems).isdisjoint(set(switches))
        components = list(gpus) + list(mems) + list(switches)
        capabilities = {}
        for g in gpus:
            capabilities[g] = {"gpu", "storage", "copy", "reduce"}
        for m in mems:
            capabilities[m] = {"storage", "copy"}
        for s in switches:
            capabilities[s] = set()
        return cls(components, capabilities, nics, edge_data, bandwidth_constraints)

    def add_ECs(self, ECs):
        # Checking that ec gpus are a partition
        flat_ec_list = [gpu for ec in ECs for gpu in ec.gpus]
        for gpu in self.gpus:
            assert gpu in flat_ec_list
        assert len(set(flat_ec_list)) == sum(len(ec.gpus) for ec in ECs)

        self.ECs = ECs

    def construct(self, T):
        """
        Generate a graph based on this topology with T time steps.
        """
        g = nx.DiGraph()
        for t in range(T + 1):
            for i in self.components:
                g.add_node((i, t))

        el = self.edge_layers

        def active(edge, t):
            layers = el.get(edge)
            return layers is None or t in layers

        for t in range(T):
            for edge in self.nnpairs:
                if active(edge, t):
                    g.add_edge((edge[0], t), (edge[1], t + 1))
            for edge in self.nspairs:
                if active(edge, t):
                    g.add_edge((edge[0], t), (edge[1], t))
            for edge in self.snpairs:
                if active(edge, t):
                    g.add_edge((edge[0], t), (edge[1], t + 1))
            for edge in self.sspairs:
                if active(edge, t):
                    g.add_edge((edge[0], t), (edge[1], t))
        return g

    def is_self_reflective(self):
        """Whether reversing every edge and swapping the copy/reduce capabilities
        yields this same topology -- i.e. a ReduceScatter here is the exact
        time-reverse of an AllGather."""
        if self.copy_nodes != self.reduce_nodes:
            return False
        for (u, v), cost in self.edge_data.items():
            if self.edge_data.get((v, u)) != cost:
                return False
        fwd = Counter(
            (frozenset(bc.edges), bc.bound) for bc in self.bandwidth_constraints
        )
        rev = Counter(
            (frozenset((v, u) for (u, v) in bc.edges), bc.bound)
            for bc in self.bandwidth_constraints
        )
        return fwd == rev


def _problem_from_topology(top: Topology):
    node_type = {c: tuple(sorted(top.capabilities[c])) for c in top.components}
    return symmetry.build_problem(
        list(top.components),
        node_type,
        top.edge_data,
        top.bandwidth_constraints,
        seed_pool=top.gpus,
    )


def generate_ecs(top):
    p = _problem_from_topology(top)
    ecs = []
    for members, rows in symmetry.compute_equivalence_classes(p):

        def fn(node, shift, members=members, rows=rows):
            return rows[shift % len(members)][node]

        ecs.append(EC(members, fn))
    return ecs


def validate_ecs(top: Topology):
    assert top.ECs is not None, "topology has no ECs"
    p = _problem_from_topology(top)
    all_nodes = set(p.nodes)

    members_all = [g for ec in top.ECs for g in ec.gpus]
    assert set(members_all) == set(top.gpus), "ECs do not cover the GPUs"
    assert len(members_all) == len(set(members_all)), "ECs overlap"

    for ec in top.ECs:
        m = len(ec.gpus)
        for shift in range(m):
            perm = {v: ec.shift_fn(v, shift) for v in p.nodes}
            assert set(perm.values()) == all_nodes, (
                "shift map is not a bijection over all nodes"
            )
            if shift == 0:
                assert all(perm[v] == v for v in p.nodes), "shift 0 is not the identity"
            assert symmetry.verify_full(p, perm), (
                "shift map is not a topology automorphism"
            )
        # Transversal contract: the shifts map the representative onto every
        # class member exactly once.
        rep = ec.gpus[0]
        images = {ec.shift_fn(rep, k) for k in range(m)}
        assert images == set(ec.gpus), "shifts do not form a transversal of the class"


def generate_multirail_from_topology(top: Topology, N, connection_bw=None):
    components = []
    capabilities = {}
    nics = []

    edge_data = {}
    bcs = []

    # First create multirail switches (pure pass-through, no capabilities).
    for nic in top.nics:
        rail_node = ("rail", nic)
        nics.append(rail_node)
        components.append(rail_node)
        capabilities[rail_node] = set()

    # Create the many copies of the topology
    for i in range(N):
        for comp in top.components:
            new_comp = (i, comp)
            components.append(new_comp)
            capabilities[new_comp] = set(top.capabilities[comp])

        for edge, data in top.edge_data.items():
            edge_data[(i, edge[0]), (i, edge[1])] = data

        for bc in top.bandwidth_constraints:
            bcs.append(
                BandwidthConstraint(
                    bc.name,
                    [((i, edge[0]), (i, edge[1])) for edge in bc.edges],
                    bc.bound,
                )
            )

        for nic in top.nics:
            edge_data[((i, nic), ("rail", nic))] = 0
            edge_data[(("rail", nic), (i, nic))] = 0

            if connection_bw is not None:
                bcs.append(
                    BandwidthConstraint(
                        f"connection {i} {nic}",
                        [((i, nic), ("rail", nic))],
                        connection_bw,
                    )
                )
                bcs.append(
                    BandwidthConstraint(
                        f"connection {i} {nic}",
                        [(("rail", nic), (i, nic))],
                        connection_bw,
                    )
                )

    new_top = Topology(components, capabilities, nics, edge_data, bcs)

    new_ECs = []

    if top.ECs is not None:
        for ec in top.ECs:
            # default-arg binding captures this class's ec (not the last loop
            # iteration's) -- same pattern as generate_ecs.
            def fn(node, shift, ec=ec):
                modulus = len(ec.gpus)
                if node[0] == "rail":
                    new_nic = ec.shift_fn(node[1], shift % modulus)
                    return ("rail", new_nic)
                else:
                    node_offset = shift // modulus
                    intra_node_shift = ec.shift_fn(node[1], shift % modulus)
                    return ((node[0] + node_offset) % N, intra_node_shift)

            new_gpus = []
            for i in range(N):
                for gpu in ec.gpus:
                    new_gpus.append((i, gpu))
            new_ECs.append(EC(new_gpus, fn))

    new_top.add_ECs(new_ECs)

    return new_top
