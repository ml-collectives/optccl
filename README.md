# OptCCL

OptCCL is a tool for generating optimal collective communication algorithms using linear programming methods. It reframes the problem as a routing problem using a multicommodity flow formulation. To solve the LPs for large topologies, OptCCL uses a technique called mirrored Dantzig-Wolfe, which exploits the inherent symmetry in the problem to accelerate the Dantzig-Wolfe subproblems.

## Installation

OptCCL requires Python >= 3.11 and a Gurobi license.

```bash
pip install -e .
```

Dependencies are `gurobipy`, `networkx`, and `numpy`.

## Configuration

OptCCL looks for a config file at `./optccl.toml` by default. You can also pass `--config PATH` to any command. All keys are optional — missing keys fall back to the defaults defined in `config.py` (see [optccl.example.toml](optccl.example.toml) for the full annotated list). To use a Gurobi WLS license, copy `optccl.example.toml` to `./optccl.toml` and fill in your credentials under `[gurobi]`; without WLS credentials, OptCCL falls back to an ordinary local Gurobi license (`gurobi.lic` on the default search path or via `GRB_LICENSE_FILE`).

The `--write-schedule` CLI flag on `process`/`run` determines if a schedule is written to file (off by default). How much of it is emitted is `expand_schedule_symmetry` under `[output]`: `false` (the default) writes the schedule for one representative origin per equivalence class and includes an `ec_expansion` block (each EC's member GPUs plus the component permutation for every shift) so the remaining members can be reconstructed; `true` writes every EC member, at the cost of the per-transmission symmetry mapping.

Set `stop_at_feasible = true` under `[solver]` to return the first feasible solution the moment Dantzig-Wolfe Phase 1 finds one, skipping the Phase 2 objective optimization entirely — this ignores solution quality in favor of runtime, useful when you just want *a* feasible collective algorithm under the given `R`.

Set `rate_retry_max` under `[solver]` (default `0`, disabled) to automatically recover from an infeasible `R` instead of erroring: when any Dantzig-Wolfe solve's Phase 1 is infeasible (the requested rate is below the collective's achievable rate on the topology), `R` is multiplied by `rate_retry_factor` (default `1.01`) and the solve is retried, up to `rate_retry_max` times. The reported `R` reflects the final (bumped) value. Applies to every DW solve, including the multi-collective stack and strengthening rounds.

## Usage

OptCCL has four top-level subcommands: `solve` (run the LP solver), `process`
(post-process a saved solver result), `run` (`solve` + `process` in one step, no
intermediate file), and `multi` (for multi-collective instances). All subcommands also accept
`-v`/`-q`/`--log-file` for logging control.

### `optccl solve` — run the LP solver

```
optccl solve <collective> <topology> <num_nodes> [-o <output.json>] [options]
```

`<collective>` is one of `all_gather`, `reduce_scatter`, `all_reduce`, `all_to_all`.

| Flag | Description |
|---|---|
| `--config PATH` | path to config file (default: `./optccl.toml`) |
| `-r`, `--rval FLOAT` | target rate R: the schedule length per unit of data (the **inverse** of throughput); auto-determined from a cut bound if omitted. If a solve reports R below the achievable rate, pass a larger value or try increasing T |
| `-T`, `--time-steps N` | number of steps in the time-expanded LP (overrides the config's `time_steps`) |
| `-o`, `--output FILE` | path to write the solver result JSON; defaults to `sol_<collective>_<topology>_<num_nodes>.json` in the current directory |
| `--strengthen` | iterative formulation strengthening (`all_gather`, `reduce_scatter`, `all_reduce`): after the solve, check that the tree decomposition recovers the full flow volume; if not, duplicate the offending vertices and re-solve, up to `strengthen_max_rounds` (`[solver]` config) times. No effect on `all_to_all` (no `w` envelope). |

Examples:

```bash
# Solve AllGather on a single DGX A100 node (NVLink only)
optccl solve all_gather a100 1 -o result.json

# Solve All-to-All on 2 DoE nodes with a custom config
optccl solve all_to_all doe 2 -o result.json --config my_config.toml
```

### `optccl process` — post-process a solver result

Decomposes a solver result into spanning trees / paths and reports the schedule's
overhead, measured latency, and algorithmic bandwidth. By default no
step-by-step schedule is materialized; pass `--write-schedule` to write the schedule, 
but note that large data sizes can lead to extremely large outputs.

```
optccl process <input.json> [-o <output.json>] [options]
```

| Flag | Description |
|---|---|
| `--config PATH` | path to config file |
| `--collective COLLECTIVE` | override the collective stored in the result file |
| `-o`, `--output FILE` | write the result JSON: the overhead/latency analysis, plus the full step schedule when `--write-schedule` is set. With `--write-schedule` and no `-o`, it defaults to `process_<input-stem>.json` next to the input |
| `--write-schedule` | run the exact WFQ and include the step-by-step schedule in `-o` (auto-named if `-o` is omitted). Coverage is set by `expand_schedule_symmetry` in the config |

Examples:

```bash
optccl process result.json                    # print the overhead/latency report
optccl process result.json -o analysis.json   # also write the analysis JSON
optccl process result.json --write-schedule -o schedule.json
```

### `optccl run` — solve and process in one step

```
optccl run <collective> <topology> <num_nodes> [options]
```

Equivalent to `optccl solve` followed by `optccl process`, keeping the solver result in
memory instead of round-tripping it through a file. It takes the union of the two
commands' options; note `-o` is the *processed* output (as in `process`), not the solver
result (as in `solve`). With `--write-schedule` and no `-o`, the schedule goes to
`process_<collective>_<topology>_<num_nodes>.json`.

```bash
optccl run all_gather a100 2 --config my_config.toml
```

### `optccl multi solve` — simultaneous multi-collective LP

Solves an arbitrary set of collectives simultaneously against shared bandwidth. The set of
collectives — each declaring which components it operates on — is authored as a **workload
spec** JSON that references a topology (a built-in name or a topology-spec path):

```
optccl multi solve <workload.json | nv | pcie > [options]
```

`nv` and `pcie` are shorthands for the bundled paper workloads under
[workloads/](workloads/) (the 8-node A100 ReduceScatter + All-to-All experiments), so the paper's
multi-collective results reproduce with e.g. `optccl multi solve nv --mode joint`.

| Flag | Description |
|---|---|
| `--mode {concat,overlay,joint}` | how to combine the collectives (default `joint`, see below) |
| `-R`, `--rval FLOAT` | target rate R for joint mode (the **inverse** of throughput; auto-determined if omitted). Ignored by concat/overlay, which auto-determine each group's own rate |
| `-T`, `--time-steps N` | number of steps in the time-expanded LP (overrides the config's `time_steps`) |
| `--validate-symmetry` | verify the demand-aware equivalence classes before solving |
| `--config PATH` | path to config file |
| `-o`, `--output FILE` | write result JSON to this file |

Modes: `joint` solves one combined LP over every collective and decomposes the demand
families (p2p paths, then gather trees, then reduce trees) against the cumulative residual
bandwidth; `overlay` solves each workload `group` separately and models them running
simultaneously (latencies add); `concat` runs the `concat_group`s one after another
(sequential phases, latencies add).

A workload spec lists collectives over explicit (post-expansion) component ids:

```json
{
  "format": "optccl-workload-spec",
  "topology": "topologies/a100_multi_nv.json",
  "num_nodes": 8,
  "collectives": [
    {"name": "a2a_node0", "type": "all_to_all", "group": "aa",
     "participants": [[0, ["g", 1]], [0, ["g", 2]]], "demand": 8},
    {"name": "x0", "type": "point_to_point", "group": "aa",
     "source": [0, ["g", 1]], "sink": [4, ["g", 1]], "demand": 16},
    {"name": "ag0", "type": "all_gather", "group": "ag", "sequential_repeats": 8,
     "participants": [[0, ["g", 1]], [1, ["g", 1]]], "demand": 16}
  ]
}
```

`type` is one of the first-class collectives `all_gather | reduce_scatter | all_to_all |
all_reduce` (taking `participants`) or a raw primitive `gather` (`source` + `sinks`),
`reduce` (`sink` + `sources`), `point_to_point` (`source` + `sink`). `topology` is a built-in
name or a path (relative to the workload file) to an `optccl-topology-spec` JSON, expanded to
`num_nodes`. `demand` is in native units (GB). `group` (default: the type) is overlay's unit
of separation; `concat_group` (default: the group) is concat's; `sequential_repeats`
(default 1, concat only) models a concat group whose solved phase runs that many times
back-to-back reduce-scatter phases.

Equivalence classes for the mirrored Dantzig-Wolfe are detected **demand-aware**: GPUs share
a class only if a topology automorphism maps one onto the other *and* maps the collective set
onto itself (collectives of equal type+demand may permute among themselves).

When `-R` is omitted, the target throughput comes from a static flow LP that does not model
copy fan-out or reduce fan-in sharing, so it is exact for p2p-dominated workloads (it
reproduces the paper's AG+AA targets) but conservative for reduce-heavy ones — pass `-R` to
override.

## Built-in topologies

| Name | Description |
|---|---|
| `doe` | DoE topology — NVLink only |
| `doe_pcie` | DoE topology — PCIe only |
| `doe_both` | DoE topology — NVLink + PCIe |
| `a100` | DGX A100 — NVLink only |
| `a100_pcie` | DGX A100 — PCIe only |
| `a100_both` | DGX A100 — NVLink + PCIe |

For multi-node runs, OptCCL automatically expands the single-node topology into a multi-rail topology using `multirail_bandwidth` as the inter-node link bandwidth. Generated topology graphs are cached to disk under `topology_cache_dir` (default `./topology_cache`) as `topology_<name>_<nodes>_<bw>.json` so subsequent runs skip the generation step.

### Custom topologies

Instead of the built-in names, the `TOPOLOGY` argument to `optccl solve` may be a path to a JSON topology spec:

```bash
optccl solve all_gather my_topology.json 1 -o out.json
optccl solve all_gather my_topology.json 4 -o out.json   # expanded to 4 nodes
```

A spec describes a **single-node** topology as a flat list of *components*, each declaring a set of **capabilities**:

| Capability | Meaning |
|---|---|
| `gpu` | source/sink of collectives (a demand endpoint) — implies all of the below |
| `storage` | occupies a layer in the time-expanded network (data can persist across a timestep) |
| `copy` | may fan-out / duplicate data (AllGather) |
| `reduce` | may fan-in / combine data (ReduceScatter/AllReduce) |

A component with no capabilities is a pure pass-through switch. Declaring `gpu` auto-fills `storage`/`copy`/`reduce`. A zero-cost persistence self-loop is added automatically for every `storage` component.

```json
{
  "format": "optccl-topology-spec",
  "components": [
    {"id": ["g", 1], "capabilities": ["gpu"]},
    {"id": ["g", 2], "capabilities": ["gpu"]},
    {"id": ["nv", 1], "capabilities": []},
    {"id": ["m", 1], "capabilities": ["storage", "copy"]},
    {"id": ["n", 1], "capabilities": []}
  ],
  "nics": [["n", 1]],
  "edges": [
    {"src": ["g", 1], "dst": ["nv", 1], "cost": 1},
    {"src": ["nv", 1], "dst": ["g", 1], "cost": 1}
  ],
  "bandwidth_constraints": [
    {"name": "nv", "edges": [[["g", 1], ["nv", 1]]], "bound": 50}
  ],
  "expansion": {"type": "multirail", "params": {"connection_bw": 25}}
}
```

Component ids may be strings, numbers, or `[type, index]` arrays. The optional `expansion` block selects the multi-node expansion applied when `num_nodes > 1` (currently only `multirail`; `connection_bw` defaults to `multirail_bandwidth`).

Alternatively, topologies can still be added in Python via [optccl/top_specs.py](optccl/top_specs.py) — using `Topology.from_node_types(...)` (or the capability-based `Topology(...)` constructor) and `BandwidthConstraint` from [optccl/topologies.py](optccl/topologies.py) — then registered in the `TOPOLOGIES` dict in [optccl/topology_spec.py](optccl/topology_spec.py).

## Module overview

| Module | Description |
|---|---|
| [optccl/topologies.py](optccl/topologies.py) | Topology data structures and multi-node expansion |
| [optccl/symmetry.py](optccl/symmetry.py) | Symmetry-detection used to build equivalence classes |
| [optccl/top_specs.py](optccl/top_specs.py) | Built-in topology definitions |
| [optccl/topology_spec.py](optccl/topology_spec.py) | Loader for human-authored JSON topology specs + multi-node expansion registry |
| [optccl/solver.py](optccl/solver.py) | Orchestrates the solve pipeline (formulation → DW → tree decomposition → strengthening) |
| [optccl/dw.py](optccl/dw.py) | Shared mirrored Dantzig-Wolfe engine (two-phase column generation over subproblems) |
| [optccl/all_gather.py](optccl/all_gather.py) | AllGather formulation |
| [optccl/reduce_scatter.py](optccl/reduce_scatter.py) | ReduceScatter formulation |
| [optccl/all_to_all.py](optccl/all_to_all.py) | All-to-All formulation |
| [optccl/all_reduce.py](optccl/all_reduce.py) | AllReduce formulation |
| [optccl/tree_decomposer.py](optccl/tree_decomposer.py) | Decompose fractional flow solutions into spanning trees / paths |
| [optccl/strengthen.py](optccl/strengthen.py) | Formulation strengthening via vertex duplication |
| [optccl/process_results.py](optccl/process_results.py) | Post-processing pipeline |
| [optccl/wfq.py](optccl/wfq.py) | Weighted fair queueing |
| [optccl/serialize.py](optccl/serialize.py) | JSON serialization/deserialization |
| [optccl/config.py](optccl/config.py) | Config loading |
| [optccl/logging_setup.py](optccl/logging_setup.py) | Logging configuration |
| [optccl/errors.py](optccl/errors.py) | User-facing error types |
| [optccl/multi/](optccl/multi/) | Multi-collective stack |
| [workloads/](workloads/) | Bundled workload specs and their topology specs |
| [optccl/cli.py](optccl/cli.py) | CLI entry point |
