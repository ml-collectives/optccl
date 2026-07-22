import json
import logging

from .errors import OptcclError
from .topologies import Topology, EC, BandwidthConstraint

logger = logging.getLogger(__name__)


def _ser_node(node):
    if isinstance(node, tuple):
        return [_ser_node(x) for x in node]
    return node


def _des_node(data):
    if isinstance(data, list):
        return tuple(_des_node(x) for x in data)
    return data


def _key_str(key):
    return json.dumps(_ser_node(key), separators=(",", ":"))


def _parse_key(s):
    return _des_node(json.loads(s))


def serialize_flow_dict(d: dict) -> dict:
    # Drop numerically-zero entries (LP solutions carry huge numbers of ~1e-15
    # values that only bloat the result file; 1e-12 is far below the pipeline's
    # decomposition tolerances).
    return {_key_str(k): v for k, v in d.items() if abs(v) > 1e-12}


def deserialize_flow_dict(d: dict) -> dict:
    return {_parse_key(k): v for k, v in d.items()}


def serialize_topology(top: Topology) -> dict:
    data = {
        "format": "optccl-topology-cache",
        "components": [_ser_node(n) for n in top.components],
        "capabilities": {
            _key_str(n): sorted(top.capabilities[n]) for n in top.components
        },
        "nics": [_ser_node(n) for n in top.nics],
        "edge_data": {_key_str(k): v for k, v in top.edge_data.items()},
        "bandwidth_constraints": [
            {
                "name": bc.name,
                "edges": [[_ser_node(e[0]), _ser_node(e[1])] for e in bc.edges],
                "bound": bc.bound,
            }
            for bc in top.bandwidth_constraints
        ],
        "ECs": None,
    }
    if top.dest_filter:
        data["dest_filter"] = {
            _key_str(k): [_ser_node(d) for d in sorted(v, key=repr)]
            for k, v in top.dest_filter.items()
        }
    if top.edge_layers:
        data["edge_layers"] = {
            _key_str(k): sorted(v) for k, v in top.edge_layers.items()
        }
    if top.ECs is not None:
        all_nodes = list(top.components)
        ecs_out = []
        for ec in top.ECs:
            n_shifts = len(ec.gpus)
            shift_table = {}
            for node in all_nodes:
                for shift in range(n_shifts):
                    try:
                        mapped = ec.shift_fn(node, shift)
                        shift_table[_key_str((node, shift))] = _ser_node(mapped)
                    except (KeyError, TypeError, IndexError):
                        pass
            ecs_out.append(
                {
                    "gpus": [_ser_node(g) for g in ec.gpus],
                    "shift_table": shift_table,
                }
            )
        data["ECs"] = ecs_out
    return data


def deserialize_topology(data: dict) -> Topology:
    components = [_des_node(n) for n in data["components"]]
    capabilities = {_parse_key(k): set(v) for k, v in data["capabilities"].items()}
    nics = [_des_node(n) for n in data["nics"]]
    edge_data = {_parse_key(k): v for k, v in data["edge_data"].items()}
    bcs = []
    for bc in data["bandwidth_constraints"]:
        edges = [(_des_node(e[0]), _des_node(e[1])) for e in bc["edges"]]
        bcs.append(BandwidthConstraint(bc["name"], edges, bc["bound"]))

    top = Topology(components, capabilities, nics, edge_data, bcs)

    if data.get("dest_filter"):
        top.dest_filter = {
            _parse_key(k): frozenset(_des_node(d) for d in v)
            for k, v in data["dest_filter"].items()
        }
    if data.get("edge_layers"):
        top.edge_layers = {
            _parse_key(k): frozenset(v) for k, v in data["edge_layers"].items()
        }

    if data.get("ECs"):
        ecs = []
        for ec_data in data["ECs"]:
            ec_gpus = [_des_node(g) for g in ec_data["gpus"]]
            table = {
                _parse_key(k): _des_node(v) for k, v in ec_data["shift_table"].items()
            }

            def _make_fn(t):
                return lambda node, shift: t[(node, shift)]

            ecs.append(EC(ec_gpus, _make_fn(table)))
        top.add_ECs(ecs)

    return top


def save_topology(path: str, top: Topology):
    with open(path, "w") as fh:
        json.dump(serialize_topology(top), fh)


def load_topology(path: str) -> Topology:
    with open(path) as fh:
        return deserialize_topology(json.load(fh))


def save_solve_result(path: str, metadata: dict, top: Topology, flows: dict):
    data = {
        "metadata": metadata,
        "topology": serialize_topology(top),
    }
    for name, flow_dict in flows.items():
        data[name] = serialize_flow_dict(flow_dict)
    with open(path, "w") as fh:
        json.dump(data, fh)
    logger.info("Saved solver result to %s", path)


def load_solve_result(path: str):
    with open(path) as fh:
        data = json.load(fh)
    metadata = data["metadata"]
    top = deserialize_topology(data["topology"])
    solver_type = metadata["solver_type"]
    flows = {}
    if solver_type == "tree":
        flows["f"] = deserialize_flow_dict(data["f"])
        flows["w"] = deserialize_flow_dict(data["w"])
    elif solver_type == "tree_paired":
        flows["f1"] = deserialize_flow_dict(data["f1"])
        flows["w1"] = deserialize_flow_dict(data["w1"])
        flows["f2"] = deserialize_flow_dict(data["f2"])
        flows["w2"] = deserialize_flow_dict(data["w2"])
    elif solver_type == "path":
        flows["f"] = deserialize_flow_dict(data["f"])
    else:
        raise OptcclError(
            f"{path}: unrecognized solver_type {solver_type!r} in the result's "
            f"metadata (expected 'tree', 'tree_paired', or 'path') -- is this a "
            f"file produced by `optccl solve`?"
        )
    return metadata, top, flows
