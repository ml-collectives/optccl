# Tool for visualizing algorithms

Renders an OptCCL schedule as a self-contained interactive HTML page: the
topology with data animated across it, a time scrubber, and one panel per run
so configurations can be compared side by side.

Works on any topology — node positions come from networkx, nothing is hardcoded.

## Example

```shell
optccl run all_gather tools/viz/examples/ring_plain.json 1 --write-schedule -o plain.json
optccl run all_gather tools/viz/examples/ring_mcast.json 1 --write-schedule -o mcast.json

python tools/viz/visualize_schedule.py plain.json mcast.json \
    --topology tools/viz/examples/ring_plain.json \
    --label "Pass-through" --label "Multicast" -o cmp.html
```

`--topology` only supplies the layout, so one spec can back every panel as long
as the runs share a topology. Built-in names work too (`--topology a100`).

## Reading the page

| | |
|---|---|
| orange link | direct GPU-to-GPU hop |
| teal link | routed through a switch |
| thickness, dot count | data volume |
| dot speed | fixed — it's the link rate, so a busier link shows *more* dots, not faster ones |
| double-outlined box | component that can `copy`/`reduce` |

Playback runs in schedule time, so longer steps genuinely take longer. Runs of
different lengths finish at different moments; the readout tracks how many are
still going.

## Examples

`ring_plain` / `ring_mcast` / `ring_nosw` are the same 4-GPU ring with a central
switch that is respectively a pass-through, multicast-capable, and absent
(a failed switch). Useful as a three-way comparison.

## Caveat

Set `n_workers = 1` in `optccl.toml`. The default parallel tree decomposition
can emit a schedule that doesn't match the objective the solver reports — on
`ring_mcast` it reports 13.333 link-hops while the schedule actually moves
16.000, silently discarding the switch fan-out. Serial decomposition is
consistent, and also removes run-to-run variance.

Schedules cover one equivalence-class representative by default, so per-link
load reads low by the size of the class. Set `expand_schedule_symmetry = true`
under `[output]` if that matters.
