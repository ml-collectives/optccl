import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

from ..errors import OptcclError
from ..serialize import _des_node
from ..topologies import Topology, generate_multirail_from_topology
from ..topology_spec import TOPOLOGIES, load_topology_spec, apply_expansion

WORKLOAD_FORMAT = "optccl-workload-spec"

UNIFORM_KINDS = ("all_gather", "reduce_scatter", "all_to_all", "all_reduce")
RAW_KINDS = ("gather", "reduce", "point_to_point")


class Gather:
    def __init__(self, source_node, sink_nodes, demand):
        self.source = source_node
        self.sinks = sink_nodes
        self.demand = demand


class Reduce:
    def __init__(self, sink_node, source_nodes, demand):
        self.sink = sink_node
        self.sources = source_nodes
        self.demand = demand


class PointToPoint:
    def __init__(self, source_node, sink_node, demand):
        self.source = source_node
        self.sink = sink_node
        self.demand = demand


@dataclass
class CollectiveEntry:
    """One parsed workload-spec entry (pre-lowering)."""

    name: str
    kind: str
    demand: float
    group: str
    concat_group: str
    participants: list = None
    root: object = None
    leaves: list = field(default_factory=list)
    sequential_repeats: int = 1


class Workload:
    """A capability ``Topology`` plus the demand set of a set of collectives."""

    def __init__(self, top: Topology, name=None):
        self.top = top
        self.name = name
        self.collectives: list[CollectiveEntry] = []

        self.gathers: list[Gather] = []
        self.reduces: list[Reduce] = []
        self.p2ps: list[PointToPoint] = []
        self.gather_by_source = defaultdict(list)
        self.reduce_by_sink = defaultdict(list)
        self.p2p_by_source = defaultdict(list)

        self.effective_gpus: set = set()
        self.ECs = None
        self._gpu_set = set(top.gpus)

        # Topology.construct only persists data at storage nodes whose (i,i)
        # self-loop is in edge_data (the retired DemandTopology force-added
        # them); spec-built topologies auto-add these, hand-built ones must too.
        for n in top.storage_nodes:
            if (n, n) not in top.edge_data:
                raise OptcclError(
                    f"storage node {n!r} lacks a persistence self-loop "
                    f"(add a zero-cost ({n!r}, {n!r}) edge)"
                )

    # --- delegation to the wrapped Topology (what the LP builders read) ---
    @property
    def copy_nodes(self):
        return self.top.copy_nodes

    @property
    def reduce_nodes(self):
        return self.top.reduce_nodes

    @property
    def edge_data(self):
        return self.top.edge_data

    @property
    def bandwidth_constraints(self):
        return self.top.bandwidth_constraints

    def construct(self, T):
        return self.top.construct(T)

    # --- demand mutators ---
    def _check_endpoint(self, node):
        if node not in self._gpu_set:
            raise OptcclError(
                f"demand endpoint {node!r} is not a gpu component of the "
                f"topology (check the participant/source/sink ids against the "
                f"expanded topology's components)"
            )

    def add_gather(self, source_node, sink_nodes, demand):
        self._check_endpoint(source_node)
        for node in sink_nodes:
            self._check_endpoint(node)
        new_gather = Gather(source_node, sink_nodes, demand)
        self.gathers.append(new_gather)
        self.effective_gpus.add(source_node)
        self.effective_gpus.update(sink_nodes)
        self.gather_by_source[source_node].append(new_gather)

    def add_reduce(self, sink_node, source_nodes, demand):
        self._check_endpoint(sink_node)
        for node in source_nodes:
            self._check_endpoint(node)
        new_reduce = Reduce(sink_node, source_nodes, demand)
        self.reduces.append(new_reduce)
        self.effective_gpus.add(sink_node)
        self.effective_gpus.update(source_nodes)
        self.reduce_by_sink[sink_node].append(new_reduce)

    def add_point_to_point(self, source_node, sink_node, demand):
        self._check_endpoint(source_node)
        self._check_endpoint(sink_node)
        new_p2p = PointToPoint(source_node, sink_node, demand)
        self.p2ps.append(new_p2p)
        self.effective_gpus.add(source_node)
        self.effective_gpus.add(sink_node)
        self.p2p_by_source[source_node].append(new_p2p)

    def add_collective(self, entry: CollectiveEntry):
        """Record a parsed entry and lower it onto the demand primitives."""
        self.collectives.append(entry)
        LOWERINGS[entry.kind](self, entry)

    def add_ECs(self, ECs):
        # The classes must partition the *effective* GPUs (demand endpoints);
        # idle GPUs never appear in flows so they need no class. (This is why
        # Topology.add_ECs, which requires covering every GPU, is not used.)
        flat_ec_list = [gpu for ec in ECs for gpu in ec.gpus]
        missing = self.effective_gpus - set(flat_ec_list)
        if missing:
            raise OptcclError(
                f"equivalence classes do not cover demand endpoint(s) "
                f"{sorted(missing, key=repr)}"
            )
        if len(set(flat_ec_list)) != sum(len(ec.gpus) for ec in ECs):
            raise OptcclError("equivalence classes overlap")
        self.ECs = ECs

    def topology_view(self) -> Topology:
        self.top.ECs = self.ECs
        return self.top


def max_demand(wl) -> float:
    vals = (
        [p.demand for p in wl.p2ps]
        + [g.demand for g in wl.gathers]
        + [r.demand for r in wl.reduces]
    )
    return max(vals) if vals else 1.0


# --- lowering: first-class collectives -> demand primitives -------------------


def _lower_all_gather(wl, ce):
    for s in ce.participants:
        wl.add_gather(s, [p for p in ce.participants if p != s], ce.demand)


def _lower_reduce_scatter(wl, ce):
    for s in ce.participants:
        wl.add_reduce(s, [p for p in ce.participants if p != s], ce.demand)


def _lower_all_to_all(wl, ce):
    for a, b in itertools.permutations(ce.participants, 2):
        wl.add_point_to_point(a, b, ce.demand)


def _lower_all_reduce(wl, ce):
    _lower_reduce_scatter(wl, ce)
    _lower_all_gather(wl, ce)


def _lower_gather(wl, ce):
    wl.add_gather(ce.root, list(ce.leaves), ce.demand)


def _lower_reduce(wl, ce):
    wl.add_reduce(ce.root, list(ce.leaves), ce.demand)


def _lower_point_to_point(wl, ce):
    wl.add_point_to_point(ce.root, ce.leaves[0], ce.demand)


LOWERINGS = {
    "all_gather": _lower_all_gather,
    "reduce_scatter": _lower_reduce_scatter,
    "all_to_all": _lower_all_to_all,
    "all_reduce": _lower_all_reduce,
    "gather": _lower_gather,
    "reduce": _lower_reduce,
    "point_to_point": _lower_point_to_point,
}


# --- spec parsing --------------------------------------------------------------


def _parse_entry(data: dict, k: int) -> CollectiveEntry:
    kind = data.get("type")
    if kind not in LOWERINGS:
        raise OptcclError(
            f"collective #{k}: unknown type {kind!r} "
            f"(expected one of {sorted(LOWERINGS)})"
        )
    demand = data.get("demand")
    if not isinstance(demand, (int, float)) or demand <= 0:
        raise OptcclError(f"collective #{k} ({kind}): demand must be a positive number")
    name = data.get("name", f"{kind}_{k}")
    group = data.get("group", kind)
    concat_group = data.get("concat_group", group)
    repeats = data.get("sequential_repeats", 1)
    if not isinstance(repeats, int) or repeats < 1:
        raise OptcclError(
            f"collective {name!r}: sequential_repeats must be a positive integer"
        )

    def req(field):
        if field not in data:
            raise OptcclError(
                f"collective {name!r} ({kind}): missing required field {field!r}"
            )
        return data[field]

    if kind in UNIFORM_KINDS:
        parts = [_des_node(p) for p in req("participants")]
        if len(parts) < 2:
            raise OptcclError(f"collective {name!r}: needs >= 2 participants")
        if len(set(parts)) != len(parts):
            raise OptcclError(f"collective {name!r}: duplicate participants")
        return CollectiveEntry(
            name,
            kind,
            demand,
            group,
            concat_group,
            participants=parts,
            sequential_repeats=repeats,
        )

    if kind == "gather":
        root = _des_node(req("source"))
        leaves = [_des_node(x) for x in req("sinks")]
    elif kind == "reduce":
        root = _des_node(req("sink"))
        leaves = [_des_node(x) for x in req("sources")]
    else:  # point_to_point
        root = _des_node(req("source"))
        leaves = [_des_node(req("sink"))]
    if not leaves:
        raise OptcclError(f"collective {name!r}: needs at least one endpoint")
    return CollectiveEntry(
        name,
        kind,
        demand,
        group,
        concat_group,
        root=root,
        leaves=leaves,
        sequential_repeats=repeats,
    )


def _resolve_topology(ref: str, num_nodes: int, cfg, base_dir: str) -> Topology:
    if ref in TOPOLOGIES:
        base = TOPOLOGIES[ref]()
        if num_nodes == 1:
            return base
        return generate_multirail_from_topology(
            base, num_nodes, cfg.multirail_bandwidth
        )
    path = ref if os.path.isabs(ref) else os.path.join(base_dir, ref)
    if not os.path.isfile(path):
        raise OptcclError(
            f"workload topology {ref!r}: not a built-in "
            f"({', '.join(TOPOLOGIES)}) and no file at {path}"
        )
    base, expansion = load_topology_spec(path)
    return apply_expansion(base, num_nodes, expansion, cfg)


def load_workload_spec(path: str, cfg, generate_ecs: bool = True) -> Workload:
    with open(path) as fh:
        data = json.load(fh)
    fmt = data.get("format")
    if fmt != WORKLOAD_FORMAT:
        raise OptcclError(
            f"{path}: not a workload spec (format={fmt!r}, "
            f"expected {WORKLOAD_FORMAT!r})"
        )

    if "topology" not in data:
        raise OptcclError(
            f"{path}: workload spec is missing the required 'topology' field"
        )
    num_nodes = data.get("num_nodes", 1)
    top = _resolve_topology(
        data["topology"],
        num_nodes,
        cfg,
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
    wl = Workload(top, name=os.path.splitext(os.path.basename(path))[0])

    names = set()
    for k, entry_data in enumerate(data.get("collectives", ())):
        entry = _parse_entry(entry_data, k)
        if entry.name in names:
            raise OptcclError(f"duplicate collective name {entry.name!r}")
        names.add(entry.name)
        wl.add_collective(entry)
    if not wl.collectives:
        raise OptcclError(f"{path}: workload has no collectives")

    if generate_ecs:
        from .symmetry import generate_workload_ecs

        wl.add_ECs(generate_workload_ecs(wl))
    return wl


def subworkload(
    wl: Workload, entries, name=None, generate_ecs: bool = True
) -> Workload:
    sub = Workload(wl.top, name=name if name is not None else wl.name)
    for ce in entries:
        sub.add_collective(ce)
    if generate_ecs:
        from .symmetry import generate_workload_ecs

        sub.add_ECs(generate_workload_ecs(sub))
    return sub
