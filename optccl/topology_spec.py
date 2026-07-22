import hashlib
import json
import logging
import os

from . import top_specs
from .errors import OptcclError
from .serialize import _des_node, deserialize_topology, load_topology, save_topology
from .topologies import (
    BandwidthConstraint,
    Topology,
    generate_ecs,
    generate_multirail_from_topology,
)

logger = logging.getLogger(__name__)

SPEC_FORMAT = "optccl-topology-spec"

TOPOLOGIES = {
    "doe_both": top_specs.gen_doe,
    "doe_pcie": top_specs.gen_doe_no_nv,
    "doe": top_specs.gen_doe_only_nv,
    "a100_both": top_specs.gen_a100,
    "a100_pcie": top_specs.gen_a100_no_nv,
    "a100": top_specs.gen_a100_only_nv,
}

_TYPE_CAPABILITIES = {
    "gpu": {"gpu", "storage", "copy", "reduce"},
    "mem": {"storage", "copy"},
    "switch": set(),
}


def _component_capabilities(comp: dict) -> set:
    if "type" in comp and "capabilities" in comp:
        raise OptcclError(
            f"component {comp.get('id')!r} sets both 'type' and 'capabilities'"
        )
    if "type" in comp:
        t = comp["type"]
        if t not in _TYPE_CAPABILITIES:
            raise OptcclError(
                f"unknown component type {t!r} (expected one of {sorted(_TYPE_CAPABILITIES)})"
            )
        return set(_TYPE_CAPABILITIES[t])
    return set(comp.get("capabilities", ()))


def _multirail_adapter(base_top, num_nodes, params, cfg):
    connection_bw = params.get("connection_bw")
    if connection_bw is None:
        connection_bw = cfg.multirail_bandwidth
    return generate_multirail_from_topology(base_top, num_nodes, connection_bw)


#: Registry of multi-node expansions. New expanders (e.g. leaf_spine, fat_tree)
#: register here as ``name -> (base_top, num_nodes, params, cfg) -> Topology``.
EXPANSIONS = {
    "multirail": _multirail_adapter,
}


def build_base_topology(data: dict) -> Topology:
    """Build the single-node base ``Topology`` from a parsed spec dict."""
    components = []
    capabilities = {}
    for comp in data["components"]:
        cid = _des_node(comp["id"])
        components.append(cid)
        capabilities[cid] = _component_capabilities(comp)

    nics = [_des_node(n) for n in data.get("nics", [])]

    edge_data = {}
    for e in data.get("edges", []):
        edge_data[(_des_node(e["src"]), _des_node(e["dst"]))] = e["cost"]

    # Persistence self-loop for storage nodes: lets data sit at a storage
    # component across a timestep for free, without the author listing (i,i).
    for cid in components:
        caps = capabilities[cid]
        if ("storage" in caps or "gpu" in caps) and (cid, cid) not in edge_data:
            edge_data[(cid, cid)] = 0

    bcs = []
    for bc in data.get("bandwidth_constraints", []):
        edges = [(_des_node(e[0]), _des_node(e[1])) for e in bc["edges"]]
        bcs.append(BandwidthConstraint(bc["name"], edges, bc["bound"]))

    top = Topology(components, capabilities, nics, edge_data, bcs)
    top.add_ECs(generate_ecs(top))
    return top


def load_topology_spec(path: str):
    """Load a JSON topology spec."""
    with open(path) as fh:
        data = json.load(fh)
    fmt = data.get("format")
    if fmt is not None and fmt != SPEC_FORMAT:
        raise OptcclError(
            f"{path}: not a topology spec (format={fmt!r}, expected {SPEC_FORMAT!r})"
        )
    try:
        base_top = build_base_topology(data)
    except KeyError as e:
        raise OptcclError(f"{path}: invalid topology spec (missing key {e})") from e
    return base_top, data.get("expansion")


def apply_expansion(base_top, num_nodes, expansion, cfg):
    """Expand a single-node base topology to ``num_nodes`` nodes."""
    if num_nodes == 1:
        return base_top
    expansion = expansion or {"type": "multirail", "params": {}}
    etype = expansion.get("type", "multirail")
    if etype not in EXPANSIONS:
        raise OptcclError(
            f"unknown expansion type {etype!r} (available: {sorted(EXPANSIONS)})"
        )
    params = expansion.get("params", {})
    return EXPANSIONS[etype](base_top, num_nodes, params, cfg)


def _topology_cache_path(
    cache_dir: str, cache_key: str, num_nodes: int, multirail_bw: float
) -> str:
    return os.path.join(
        cache_dir, f"topology_{cache_key}_{num_nodes}_{multirail_bw}.json"
    )


def load_or_build_topology(topology_name: str, num_nodes: int, cfg):
    """Resolve a topology from either a built-in name or a JSON spec/cache file."""
    multirail_bw = cfg.multirail_bandwidth
    is_builtin = topology_name in TOPOLOGIES
    is_file = os.path.isfile(topology_name)
    if not is_builtin and not is_file:
        raise OptcclError(
            f"unknown topology '{topology_name}'. Available built-ins: "
            f"{', '.join(TOPOLOGIES)} (or pass a path to a .json topology spec)."
        )

    data = None
    if is_builtin:
        cache_key = topology_name
    else:
        with open(topology_name, "rb") as fh:
            raw = fh.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise OptcclError(f"{topology_name}: invalid JSON ({e})")
        stem = os.path.splitext(os.path.basename(topology_name))[0]
        cache_key = f"{stem}-{hashlib.sha256(raw).hexdigest()[:10]}"

    cache_dir = cfg.topology_cache_dir
    cache_path = _topology_cache_path(cache_dir, cache_key, num_nodes, multirail_bw)
    if os.path.exists(cache_path):
        logger.info(
            "Loading cached topology from %s (delete it to force a rebuild)", cache_path
        )
        return load_topology(cache_path)

    if is_builtin:
        base_top = TOPOLOGIES[topology_name]()
        top = (
            base_top
            if num_nodes == 1
            else generate_multirail_from_topology(base_top, num_nodes, multirail_bw)
        )
    elif data.get("format") == SPEC_FORMAT:
        try:
            base_top = build_base_topology(data)
        except KeyError as e:
            raise OptcclError(
                f"{topology_name}: invalid topology spec (missing key {e})"
            ) from e
        top = apply_expansion(base_top, num_nodes, data.get("expansion"), cfg)
    else:
        # A cache dump (or a legacy/untagged topology JSON) -- load as a full
        # topology.
        try:
            top = deserialize_topology(data)
        except KeyError as e:
            raise OptcclError(
                f'{topology_name}: no "format": "{SPEC_FORMAT}" tag and it does '
                f"not parse as a topology cache either (missing key {e})"
            ) from e

    os.makedirs(cache_dir, exist_ok=True)
    save_topology(cache_path, top)
    logger.info("Saved topology to %s", cache_path)
    return top
