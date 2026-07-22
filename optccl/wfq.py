import logging
import math
import heapq
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


def physical_edges(spatial_edge, switches):
    """Expand an elided logical edge (u, v) into its true physical sub-edges,
    given the ordered list of switches the flow actually passes through
    between u and v. With no switches, the logical edge is itself physical."""
    u, v = spatial_edge
    if not switches:
        return [(u, v)]
    hops = [(u, switches[0])]
    for i in range(len(switches) - 1):
        hops.append((switches[i], switches[i + 1]))
    hops.append((switches[-1], v))
    return hops


def _admit_transmissions(edge_queues, get_phys_edges, capacity_phys):
    remaining = defaultdict(int)
    frontier = []
    for se, q in edge_queues.items():
        if q:
            vft, tid, chunk_id, lane_id = q[0]
            heapq.heappush(frontier, (vft, se, tid, chunk_id, lane_id))

    admitted = []
    blocked_ses = set()
    while frontier:
        vft, se, tid, chunk_id, lane_id = heapq.heappop(frontier)
        if se in blocked_ses:
            continue
        pes = get_phys_edges(tid, lane_id)
        if all(remaining[pe] < capacity_phys.get(pe, 0) for pe in pes):
            heapq.heappop(edge_queues[se])
            for pe in pes:
                remaining[pe] += 1
            admitted.append((se, vft, tid, chunk_id, lane_id))
            q = edge_queues[se]
            if q:
                nvft, ntid, nchunk_id, nlane_id = q[0]
                heapq.heappush(frontier, (nvft, se, ntid, nchunk_id, nlane_id))
        else:
            blocked_ses.add(se)
    return admitted


def _admit_transmissions_credits(
    edge_queues, get_phys_edges, rate_phys, credits, bucket
):
    """Token-bucket variant of _admit_transmissions."""
    for pe, r in rate_phys.items():
        credits[pe] = min(credits[pe] + r, bucket.get(pe, r))

    frontier = []
    for se, q in edge_queues.items():
        if q:
            vft, tid, chunk_id, lane_id = q[0]
            heapq.heappush(frontier, (vft, se, tid, chunk_id, lane_id))

    admitted = []
    blocked_ses = set()
    while frontier:
        vft, se, tid, chunk_id, lane_id = heapq.heappop(frontier)
        if se in blocked_ses:
            continue
        pes = get_phys_edges(tid, lane_id)
        if all(credits[pe] >= 1.0 for pe in pes):
            heapq.heappop(edge_queues[se])
            for pe in pes:
                credits[pe] -= 1.0
            admitted.append((se, vft, tid, chunk_id, lane_id))
            q = edge_queues[se]
            if q:
                nvft, ntid, nchunk_id, nlane_id = q[0]
                heapq.heappush(frontier, (nvft, se, ntid, nchunk_id, nlane_id))
        else:
            blocked_ses.add(se)
    return admitted


def build_tree_dag(edge_list):
    lanes = {}
    for idx, (i, j, t, meta) in enumerate(edge_list):
        lanes[idx] = {
            "src": i,
            "dst": j,
            "time": t,
            "spatial": (i, j),
            "metadata": meta if meta else {},
        }

    # Group edges by source spatial node and sort by time
    edges_from = defaultdict(list)
    for eid, info in lanes.items():
        edges_from[info["src"]].append((info["time"], eid, info["src"], info["dst"]))
    for node in edges_from:
        edges_from[node].sort()

    def find_real_successors(spatial_node, after_time, visited=None):
        if visited is None:
            visited = set()
        if spatial_node in visited:
            return []
        visited.add(spatial_node)
        result = []
        for t, eid, src, dst in edges_from.get(spatial_node, []):
            if t < after_time:
                continue
            if src == dst:
                # Self-loop: walk through it
                result.extend(find_real_successors(dst, t, visited))
            else:
                result.append(eid)
        return result

    # Find root
    real_targets = set()
    all_nodes = set()
    for eid, info in lanes.items():
        all_nodes.add(info["src"])
        all_nodes.add(info["dst"])
        if info["src"] != info["dst"]:
            real_targets.add(info["dst"])

    root_candidates = all_nodes - real_targets
    assert len(root_candidates) == 1, (
        f"Expected 1 root, got {len(root_candidates)}: {root_candidates}, \n\n {edge_list}"
    )
    root = root_candidates.pop()

    real_eids = [eid for eid, info in lanes.items() if info["src"] != info["dst"]]
    lane_successors = {}
    for eid in real_eids:
        info = lanes[eid]
        lane_successors[eid] = find_real_successors(info["dst"], info["time"])

    root_lanes = find_real_successors(root, -1)

    spatial_edges = set()
    for eid in real_eids:
        spatial_edges.add(lanes[eid]["spatial"])

    # Leaves
    leaves = set()
    for eid in real_eids:
        if not lane_successors.get(eid):
            leaves.add(lanes[eid]["dst"])

    return root, root_lanes, lane_successors, spatial_edges, lanes, leaves


def trim_schedule(schedule, delivered_order, required_chunks):
    # Identify which chunks to keep
    keep_chunks = set()
    cutoff_step = 0
    for i, (tid, cid, step) in enumerate(delivered_order):
        if i >= required_chunks:
            break
        keep_chunks.add((tid, cid))
        cutoff_step = max(cutoff_step, step)

    # Build trimmed schedule
    trimmed = {}
    removed_transmissions = 0
    kept_transmissions = 0

    for t in sorted(schedule.keys()):
        if t > cutoff_step:
            for ue, chunks in schedule[t].items():
                removed_transmissions += len(chunks)
            continue

        step_data = {}
        for ue, chunks in schedule[t].items():
            kept = [
                entry
                for entry in chunks
                if (entry["tree_id"], entry["chunk_id"]) in keep_chunks
            ]
            removed_transmissions += len(chunks) - len(kept)
            kept_transmissions += len(kept)
            if kept:
                step_data[ue] = kept

        if step_data:
            trimmed[t] = step_data

    logger.debug(
        "Trimmed schedule: kept %s chunks, cutoff at step %s",
        required_chunks,
        cutoff_step,
    )
    logger.debug(
        "  Transmissions: %d kept, %d removed",
        kept_transmissions,
        removed_transmissions,
    )

    return trimmed


def wfq_load_counts(trees, demand, C, K, R, bucket_extra=1.0):
    """Count-based WFQ load profile for one origin.

    Instead of moving individual chunks through the queues, each lane carries an integer
    COUNT of queued chunks and each step admits a credit-limited count per lane (the same
    token-bucket / ceil capacity the per-chunk sim uses, since admission is gated by
    credits and chunks within a lane are interchangeable for the load profile). Returns
    ``(edge_load, n_steps)`` where edge_load maps each true physical edge to {step: bytes}.
    The result feeds the same EC-symmetry expansion + analyze_bc_step_load as the exact
    streaming path.
    """
    if not trees:
        return {}, 0

    infos = []
    for tree_id, (edge_list, volume, _dests) in enumerate(trees):
        x_tau = volume * demand
        n_chunks = math.ceil(x_tau / C)
        if n_chunks == 0:
            continue
        root, root_lanes, lane_successors, _se, lanes, _leaves = build_tree_dag(
            edge_list
        )
        # Forward along a BFS spanning tree so each lane is fed exactly once per chunk
        # (the per-chunk sim enforces this via per-chunk dedup; counts have no identity,
        # so any cycle/reconvergence in lane_successors would feed a lane forever).
        fwd_tree = defaultdict(list)
        visited = set(root_lanes)
        queue = list(root_lanes)
        while queue:
            lid = queue.pop()
            for s in lane_successors.get(lid, []):
                if s not in visited:
                    visited.add(s)
                    fwd_tree[lid].append(s)
                    queue.append(s)
        infos.append(
            {
                "id": tree_id,
                "n_chunks": n_chunks,
                "rate": x_tau / (K * C),
                "root_lanes": root_lanes,
                "lane_successors": dict(fwd_tree),
                "lanes": lanes,
                "phys": {
                    lid: physical_edges(l["spatial"], l["metadata"].get("switches", []))
                    for lid, l in lanes.items()
                },
            }
        )
    if not infos:
        return {}, 0

    # Shared per-physical-edge token bucket (the real capacity).
    r_phys = defaultdict(float)
    for info in infos:
        for pes in info["phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    credits = defaultdict(float, dict(bucket))

    lane_rate = {}
    sent = {}
    lanes_flat = []  # (tree_id, lane_id)
    for info in infos:
        r = info["rate"]
        for lid in info["lanes"]:
            key = (info["id"], lid)
            lane_rate[key] = r
            sent[key] = 0.0
            lanes_flat.append(key)

    # Per-tree chunk introduction schedule (rate-paced, matches wfq_schedule).
    intro = {}  # tree_id -> {step: count}
    for info in infos:
        sched_i = defaultdict(int)
        rate, n = info["rate"], info["n_chunks"]
        done = 0
        t = 0
        while done < n:
            new = min(math.ceil(rate * (t + 1)) - math.ceil(rate * t), n - done)
            if new:
                sched_i[t] += new
            done += new
            t += 1
        intro[info["id"]] = sched_i

    # Flat per-lane lookups: physical edges (credit deduction + load) and successors.
    lane_phys = {
        (info["id"], lid): info["phys"][lid] for info in infos for lid in info["lanes"]
    }
    lane_succ = {
        (info["id"], lid): [
            (info["id"], s) for s in info["lane_successors"].get(lid, [])
        ]
        for info in infos
        for lid in info["lanes"]
    }

    # Q[(tree_id, lane_id)] = integer count of chunks queued on that lane.
    Q = defaultdict(int)
    edge_load = defaultdict(lambda: defaultdict(float))

    max_steps = K * 20 + 100
    last_intro = max((max(s) if s else 0) for s in intro.values())
    t = 0
    n_steps = 0
    while t < max_steps:
        # Refill the shared per-physical-edge token buckets.
        for pe, r in r_phys.items():
            c = credits[pe] + r
            b = bucket[pe]
            credits[pe] = c if c < b else b

        # Introduce new chunks at the trees' root lanes.
        for info in infos:
            new = intro[info["id"]].get(t, 0)
            if new:
                tid = info["id"]
                for rl in info["root_lanes"]:
                    Q[(tid, rl)] += new

        # Work-conserving fair admission: serve backlogged lanes in deficit order
        # (lowest sent/rate first ~ VFT), each greedily taking shared edge credits.
        active = [k for k in lanes_flat if Q[k]]
        active.sort(key=lambda k: sent[k] / lane_rate[k])
        forward = defaultdict(int)
        any_activity = False
        for key in active:
            q = Q[key]
            send = q
            for pe in lane_phys[key]:
                avail = int(credits[pe])  # floor: integer admits, fractional carries
                if avail < send:
                    send = avail
            if send <= 0:
                continue
            for pe in lane_phys[key]:
                credits[pe] -= send
                edge_load[pe][t] += send * C
            Q[key] = q - send
            sent[key] += send
            for s in lane_succ[key]:
                forward[s] += send
            any_activity = True

        for key, cnt in forward.items():
            Q[key] += cnt

        if any_activity:
            n_steps = t + 1
        elif t > last_intro and not any(Q.values()):
            break
        t += 1

    if any(Q.values()):
        logger.warning(
            "count-based WFQ hit its safety limit (%d steps) with chunks still "
            "queued; the load profile is truncated and reported latencies are "
            "unreliable",
            max_steps,
        )
    return dict(edge_load), n_steps


def wfq_schedule_from_counts(
    trees, demand, C, K, R, required_chunks=None, bucket_extra=1.0, path_mode=False
):
    """Materialize a step schedule from the count sim's admission decisions."""
    logger.info("WFQ (from counts)")
    infos = _build_count_infos(trees, demand, C, K)
    if not infos:
        return {}

    r_phys = defaultdict(float)
    for info in infos:
        for pes in info["phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    credits = defaultdict(float, dict(bucket))

    lane_rate, sent, flat, lane_phys, lane_succ, lane_se, lane_meta = (
        {},
        {},
        [],
        {},
        {},
        {},
        {},
    )
    for info in infos:
        dest = trees[info["id"]][2] if path_mode else None
        for lid in info["lanes"]:
            key = (info["id"], lid)
            lane_rate[key] = info["rate"]
            sent[key] = 0.0
            flat.append(key)
            lane_phys[key] = info["phys"][lid]
            lane_succ[key] = [(info["id"], s) for s in info["succ"].get(lid, [])]
            lane_se[key] = info["lanes"][lid]["spatial"]
            meta = info["lanes"][lid]["metadata"]
            lane_meta[key] = {**meta, "dest": dest} if path_mode else meta

    Qid = {key: deque() for key in flat}
    next_chunk = {info["id"]: 0 for info in infos}
    schedule = defaultdict(lambda: defaultdict(list))
    last_intro = max((max(s) if s else 0) for s in (info["intro"] for info in infos))
    max_steps = K * 20 + 100
    t = 0
    while t < max_steps:
        for pe, r in r_phys.items():
            c = credits[pe] + r
            b = bucket[pe]
            credits[pe] = c if c < b else b
        # Introduce chunks (rate-paced): each new chunk gets a fresh id, broadcast to all roots.
        for info in infos:
            new = info["intro"].get(t, 0)
            if new:
                tid = info["id"]
                for _ in range(new):
                    cid = next_chunk[tid]
                    next_chunk[tid] += 1
                    for rl in info["root_lanes"]:
                        Qid[(tid, rl)].append(cid)
        active = [k for k in flat if Qid[k]]
        active.sort(key=lambda k: sent[k] / lane_rate[k])
        forward = defaultdict(list)
        any_activity = False
        for key in active:
            send = len(Qid[key])
            for pe in lane_phys[key]:
                avail = int(credits[pe])
                if avail < send:
                    send = avail
            if send <= 0:
                continue
            for pe in lane_phys[key]:
                credits[pe] -= send
            ids = [Qid[key].popleft() for _ in range(send)]
            tid, se, meta = key[0], lane_se[key], lane_meta[key]
            bucket_list = schedule[t][se]
            for cid in ids:
                bucket_list.append({"tree_id": tid, "chunk_id": cid, "metadata": meta})
            sent[key] += send
            for s in lane_succ[key]:
                forward[s].extend(ids)
            any_activity = True
        for key, ids in forward.items():
            Qid[key].extend(ids)
        if not any_activity and t > last_intro and not any(Qid.values()):
            break
        t += 1

    if any(Qid.values()):
        logger.warning(
            "count-based WFQ hit its safety limit (%d steps) with chunks still "
            "queued; the schedule is truncated and reported latencies are "
            "unreliable",
            max_steps,
        )
    return {step: dict(edge_map) for step, edge_map in schedule.items()}


def _build_count_infos(trees, demand, C, K):
    """Per-tree count-sim state: BFS spanning tree successors (so each lane is fed once per
    chunk), physical edges, rate, chunk count, and rate-paced introduction schedule."""
    infos = []
    for tree_id, (edge_list, volume, _dests) in enumerate(trees):
        x_tau = volume * demand
        n_chunks = math.ceil(x_tau / C)
        if n_chunks == 0:
            continue
        root, root_lanes, lane_successors, _se, lanes, _leaves = build_tree_dag(
            edge_list
        )
        fwd_tree = defaultdict(list)
        visited = set(root_lanes)
        queue = list(root_lanes)
        while queue:
            lid = queue.pop()
            for s in lane_successors.get(lid, []):
                if s not in visited:
                    visited.add(s)
                    fwd_tree[lid].append(s)
                    queue.append(s)
        intro = defaultdict(int)
        rate, done, t = x_tau / (K * C), 0, 0
        while done < n_chunks:
            new = min(math.ceil(rate * (t + 1)) - math.ceil(rate * t), n_chunks - done)
            if new:
                intro[t] += new
            done += new
            t += 1
        infos.append(
            {
                "id": tree_id,
                "n_chunks": n_chunks,
                "rate": x_tau / (K * C),
                "root_lanes": root_lanes,
                "succ": dict(fwd_tree),
                "lanes": lanes,
                "intro": intro,
                # Lanes actually fed (BFS-reachable from the roots): every chunk crosses exactly
                # these, so a chunk is delivered once the slowest of them has sent it.
                "fed": set(visited),
                "phys": {
                    lid: physical_edges(l["spatial"], l["metadata"].get("switches", []))
                    for lid, l in lanes.items()
                },
            }
        )
    return infos


def _count_phase_state(infos, bucket_extra):
    r_phys = defaultdict(float)
    for info in infos:
        for pes in info["phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    lane_rate, lane_phys, lane_succ, flat = {}, {}, {}, []
    for info in infos:
        for lid in info["lanes"]:
            key = (info["id"], lid)
            lane_rate[key] = info["rate"]
            lane_phys[key] = info["phys"][lid]
            lane_succ[key] = [(info["id"], s) for s in info["succ"].get(lid, [])]
            flat.append(key)
    return {
        "r_phys": r_phys,
        "bucket": bucket,
        "credits": defaultdict(float, dict(bucket)),
        "lane_rate": lane_rate,
        "lane_phys": lane_phys,
        "lane_succ": lane_succ,
        "flat": flat,
        "Q": defaultdict(int),
        "sent": {k: 0.0 for k in flat},
    }


def _count_admit(st, t, edge_load, C):
    for pe, r in st["r_phys"].items():
        c = st["credits"][pe] + r
        b = st["bucket"][pe]
        st["credits"][pe] = c if c < b else b
    Q, sent, lane_rate = st["Q"], st["sent"], st["lane_rate"]
    active = [k for k in st["flat"] if Q[k]]
    active.sort(key=lambda k: sent[k] / lane_rate[k])
    forward = defaultdict(int)
    moved = False
    for key in active:
        q = Q[key]
        send = q
        for pe in st["lane_phys"][key]:
            avail = int(st["credits"][pe])
            if avail < send:
                send = avail
        if send <= 0:
            continue
        for pe in st["lane_phys"][key]:
            st["credits"][pe] -= send
            edge_load[pe][t] += send * C
        Q[key] = q - send
        sent[key] += send
        for s in st["lane_succ"][key]:
            forward[s] += send
        moved = True
    for key, cnt in forward.items():
        Q[key] += cnt
    return moved


def wfq_allreduce_load_counts(rs_trees, ag_trees, demand, C, K, R, bucket_extra=1.0):
    """Count-based combined reduce-scatter + all-gather load profile for all_reduce."""
    rs_infos = _build_count_infos(rs_trees, demand, C, K)
    ag_infos = _build_count_infos(ag_trees, demand, C, K)
    if not rs_infos and not ag_infos:
        return {}, 0

    rs = _count_phase_state(rs_infos, bucket_extra)
    ag = _count_phase_state(ag_infos, bucket_extra)
    rs_root = {info["id"]: info["root_lanes"] for info in rs_infos}
    ag_root = {info["id"]: info["root_lanes"] for info in ag_infos}

    # AG introduction order: tree ids rate-interleaved (matches ag_intro_order in the exact),
    # consumed one per RS completion.
    ag_order = []
    for info in ag_infos:
        rate, done, t = info["rate"], 0, 0
        while done < info["n_chunks"]:
            new = min(
                math.ceil(rate * (t + 1)) - math.ceil(rate * t), info["n_chunks"] - done
            )
            ag_order.extend([info["id"]] * new)
            done += new
            t += 1
    total_ag = len(ag_order)

    edge_load = defaultdict(lambda: defaultdict(float))
    rs_total = sum(info["n_chunks"] for info in rs_infos)
    rs_intro_done = 0
    rs_completed_prev = 0.0
    ag_intro_ptr = 0  # next index in ag_order to introduce
    ag_cleared = 0  # AG chunks cleared by RS completions, not yet introduced

    max_steps = K * 40 + 200
    t = 0
    n_steps = 0
    while t < max_steps:
        # Introduce RS chunks (rate-paced).
        for info in rs_infos:
            new = info["intro"].get(t, 0)
            if new:
                for rl in rs_root[info["id"]]:
                    rs["Q"][(info["id"], rl)] += new
                rs_intro_done += new
        # Introduce AG chunks cleared by prior RS completions.
        if ag_cleared and ag_intro_ptr < total_ag:
            take = min(ag_cleared, total_ag - ag_intro_ptr)
            per_tree = defaultdict(int)
            for tid in ag_order[ag_intro_ptr : ag_intro_ptr + take]:
                per_tree[tid] += 1
            for tid, cnt in per_tree.items():
                for rl in ag_root[tid]:
                    ag["Q"][(tid, rl)] += cnt
            ag_intro_ptr += take
            ag_cleared -= take

        rs_moved = _count_admit(rs, t, edge_load, C)
        # RS completions so far = sum over RS trees of (min cumulative sent across that
        # tree's fed lanes) -- a chunk is delivered once its slowest lane has sent it. New
        # completions clear that many AG chunks for introduction next step.
        rs_completed = 0.0
        for info in rs_infos:
            tid, fed = info["id"], info["fed"]
            if fed:
                rs_completed += min(rs["sent"][(tid, lid)] for lid in fed)
        new_comp = rs_completed - rs_completed_prev
        if new_comp > 0:
            ag_cleared += int(new_comp)
            rs_completed_prev += int(new_comp)
        ag_moved = _count_admit(ag, t, edge_load, C)

        if rs_moved or ag_moved:
            n_steps = t + 1
        elif (
            rs_intro_done >= rs_total
            and ag_intro_ptr >= total_ag
            and not any(rs["Q"].values())
            and not any(ag["Q"].values())
        ):
            break
        t += 1

    if (
        rs_intro_done < rs_total
        or ag_intro_ptr < total_ag
        or any(rs["Q"].values())
        or any(ag["Q"].values())
    ):
        logger.warning(
            "all_reduce count-based WFQ hit its safety limit (%d steps) with "
            "chunks still queued; the load profile is truncated and reported "
            "latencies are unreliable",
            max_steps,
        )
    return dict(edge_load), n_steps


def _ar_count_phase(infos, bucket_extra):
    r_phys = defaultdict(float)
    for info in infos:
        for pes in info["phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    st = {
        "r_phys": r_phys,
        "bucket": bucket,
        "credits": defaultdict(float, dict(bucket)),
        "lane_rate": {},
        "lane_phys": {},
        "lane_succ": {},
        "lane_se": {},
        "lane_meta": {},
        "flat": [],
        "sent": {},
        "Qid": {},
    }
    for info in infos:
        for lid in info["lanes"]:
            k = (info["id"], lid)
            st["lane_rate"][k] = info["rate"]
            st["sent"][k] = 0.0
            st["flat"].append(k)
            st["lane_phys"][k] = info["phys"][lid]
            st["lane_succ"][k] = [(info["id"], s) for s in info["succ"].get(lid, [])]
            st["lane_se"][k] = info["lanes"][lid]["spatial"]
            st["lane_meta"][k] = info["lanes"][lid]["metadata"]
            st["Qid"][k] = deque()
    return st


def _ar_count_admit(st, t, sched):
    for pe, r in st["r_phys"].items():
        c = st["credits"][pe] + r
        b = st["bucket"][pe]
        st["credits"][pe] = c if c < b else b
    active = [k for k in st["flat"] if st["Qid"][k]]
    active.sort(key=lambda k: st["sent"][k] / st["lane_rate"][k])
    forward = defaultdict(list)
    moved = False
    for key in active:
        send = len(st["Qid"][key])
        for pe in st["lane_phys"][key]:
            avail = int(st["credits"][pe])
            if avail < send:
                send = avail
        if send <= 0:
            continue
        for pe in st["lane_phys"][key]:
            st["credits"][pe] -= send
        ids = [st["Qid"][key].popleft() for _ in range(send)]
        tid, se, meta = key[0], st["lane_se"][key], st["lane_meta"][key]
        bucket_list = sched[t][se]
        for gid in ids:
            bucket_list.append(
                {"tree_id": tid, "chunk_id": gid, "global_id": gid, "metadata": meta}
            )
        st["sent"][key] += send
        for s in st["lane_succ"][key]:
            forward[s].extend(ids)
        moved = True
    for key, ids in forward.items():
        st["Qid"][key].extend(ids)
    return moved


def wfq_allreduce_schedule_from_counts(
    rs_trees, ag_trees, demand, C, K, R, bucket_extra=1.0
):
    """Count-based materialized schedule for paired all_reduce."""
    logger.info("WFQ AllReduce (from counts)")
    rs_infos = _build_count_infos(rs_trees, demand, C, K)
    ag_infos = _build_count_infos(ag_trees, demand, C, K)
    if not rs_infos or not ag_infos:
        return {}, {}

    rs = _ar_count_phase(rs_infos, bucket_extra)
    ag = _ar_count_phase(ag_infos, bucket_extra)
    rs_root = {info["id"]: info["root_lanes"] for info in rs_infos}
    ag_root = {info["id"]: info["root_lanes"] for info in ag_infos}

    ag_order = []
    for info in ag_infos:
        rate, done, t = info["rate"], 0, 0
        while done < info["n_chunks"]:
            new = min(
                math.ceil(rate * (t + 1)) - math.ceil(rate * t), info["n_chunks"] - done
            )
            ag_order.extend([info["id"]] * new)
            done += new
            t += 1
    total_ag = len(ag_order)

    rs_sched = defaultdict(lambda: defaultdict(list))
    ag_sched = defaultdict(lambda: defaultdict(list))
    rs_total = sum(info["n_chunks"] for info in rs_infos)
    rs_gid = ag_gid = 0  # sequential global ids per phase (intro order)
    rs_intro_done = 0
    rs_completed_prev = 0.0
    ag_ptr = 0  # next index into ag_order
    ag_cleared = 0  # AG chunks cleared by RS completions, awaiting introduction
    max_steps = K * 40 + 200
    t = 0
    while t < max_steps:
        for info in rs_infos:  # introduce RS (rate-paced)
            new = info["intro"].get(t, 0)
            if new:
                tid = info["id"]
                for _ in range(new):
                    gid = rs_gid
                    rs_gid += 1
                    for rl in rs_root[tid]:
                        rs["Qid"][(tid, rl)].append(gid)
                rs_intro_done += new
        if ag_cleared and ag_ptr < total_ag:  # introduce AG (cleared by RS completions)
            take = min(ag_cleared, total_ag - ag_ptr)
            per = defaultdict(list)
            for tid in ag_order[ag_ptr : ag_ptr + take]:
                per[tid].append(ag_gid)
                ag_gid += 1
            for tid, gids in per.items():
                for rl in ag_root[tid]:
                    ag["Qid"][(tid, rl)].extend(gids)
            ag_ptr += take
            ag_cleared -= take

        rs_moved = _ar_count_admit(rs, t, rs_sched)
        rs_completed = 0.0  # completions = sum of per-tree min-sent
        for info in rs_infos:
            tid, fed = info["id"], info["fed"]
            if fed:
                rs_completed += min(rs["sent"][(tid, lid)] for lid in fed)
        new_comp = rs_completed - rs_completed_prev
        if new_comp > 0:
            ag_cleared += int(new_comp)
            rs_completed_prev += int(new_comp)
        ag_moved = _ar_count_admit(ag, t, ag_sched)

        if (
            not rs_moved
            and not ag_moved
            and rs_intro_done >= rs_total
            and ag_ptr >= total_ag
            and not any(rs["Qid"].values())
            and not any(ag["Qid"].values())
        ):
            break
        t += 1

    if (
        rs_intro_done < rs_total
        or ag_ptr < total_ag
        or any(rs["Qid"].values())
        or any(ag["Qid"].values())
    ):
        logger.warning(
            "all_reduce count-based WFQ hit its safety limit (%d steps) with "
            "chunks still queued; the schedule is truncated and reported "
            "latencies are unreliable",
            max_steps,
        )
    return (
        {s: dict(em) for s, em in rs_sched.items()},
        {s: dict(em) for s, em in ag_sched.items()},
    )


def wfq_schedule(
    trees, demand, C, K, R, required_chunks=None, bucket_extra=1.0, emit=None
):
    """If ``emit`` is given, each admitted transmission is streamed to
    ``emit(t, se, tid, chunk_id, meta)`` instead of being stored, and an empty schedule
    is returned (no trimming)."""
    logger.info("WFQ")
    if not trees:
        return {}

    # Build tree structures
    tree_infos = []
    for tree_id, (edge_list, volume, _edge_dests) in enumerate(trees):
        x_tau = volume * demand
        n_chunks = math.ceil(x_tau / C)
        if n_chunks == 0:
            continue

        root, root_lanes, lane_successors, spatial_edges, lanes, leaves = (
            build_tree_dag(edge_list)
        )
        rate = x_tau / (K * C)

        tree_infos.append(
            {
                "id": tree_id,
                "x_tau": x_tau,
                "n_chunks": n_chunks,
                "root": root,
                "root_lanes": root_lanes,
                "lane_successors": lane_successors,
                "spatial_edges": spatial_edges,
                "lanes": lanes,
                "leaves": leaves,
                "rate": rate,
            }
        )

    if not tree_infos:
        return {}

    tree_by_id = {info["id"]: info for info in tree_infos}

    # Expand each lane's elided edge into the physical sub-edges it actually
    # crosses, and aggregate rate/capacity per physical edge so that lanes
    # sharing a switch uplink/downlink draw from the same budget.
    for info in tree_infos:
        info["lane_phys"] = {
            lane_id: physical_edges(
                lane["spatial"], lane["metadata"].get("switches", [])
            )
            for lane_id, lane in info["lanes"].items()
        }

    r_phys = defaultdict(float)
    for info in tree_infos:
        for pes in info["lane_phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]

    capacity_phys = {pe: math.ceil(r) for pe, r in r_phys.items()}
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    credits = defaultdict(float, {pe: b for pe, b in bucket.items()})

    # Precompute per-lane constants used in enqueue (the VFT increment and the spatial
    # edge are fixed per lane, so recomputing the bottleneck genexpr per chunk is wasted).
    for info in tree_infos:
        x_tau = info["x_tau"]
        vft_incr = {}
        lane_se = {}
        for lane_id, pes in info["lane_phys"].items():
            bottleneck = max(r_phys[pe] / capacity_phys[pe] for pe in pes)
            vft_incr[lane_id] = K * C * bottleneck / x_tau
            lane_se[lane_id] = info["lanes"][lane_id]["spatial"]
        info["vft_incr"] = vft_incr
        info["lane_se"] = lane_se

    # Chunk introduction schedule
    pending_intro = defaultdict(list)
    for info in tree_infos:
        tid = info["id"]
        rate = info["rate"]
        n = info["n_chunks"]
        chunks_so_far = 0
        t = 0
        while chunks_so_far < n:
            new_chunks = min(
                math.ceil(rate * (t + 1)) - math.ceil(rate * t),
                n - chunks_so_far,
            )
            for _c in range(new_chunks):
                pending_intro[t].append((tid, chunks_so_far))
                chunks_so_far += 1
            t += 1

    # WFQ simulation
    total_chunks = sum(info["n_chunks"] for info in tree_infos)
    schedule = defaultdict(lambda: defaultdict(list))

    # spatial_edge -> heap of (vft, tree_id, chunk_id, lane_id)
    edge_queues = defaultdict(list)
    v_prev = defaultdict(float)  # (tree_id, lane_id) -> last VFT
    chunk_pending = defaultdict(int)  # (tree_id, chunk_id) -> pending count
    chunk_enqueued = defaultdict(set)  # (tree_id, chunk_id) -> set of lane_ids

    def enqueue(lane_id, tree_id, chunk_id, arrival_step):
        key = (tree_id, chunk_id)
        enq = chunk_enqueued[key]
        if lane_id in enq:
            return
        enq.add(lane_id)

        info = tree_by_id[tree_id]
        vkey = (tree_id, lane_id)
        prev_vft = v_prev.get(vkey, 0)
        base = arrival_step if arrival_step > prev_vft else prev_vft
        vft = base + info["vft_incr"][lane_id]
        v_prev[vkey] = vft

        heapq.heappush(
            edge_queues[info["lane_se"][lane_id]], (vft, tree_id, chunk_id, lane_id)
        )
        chunk_pending[key] += 1

    def introduce_chunks(t):
        for tid, chunk_id in pending_intro.get(t, []):
            info = tree_by_id[tid]
            for lane_id in info["root_lanes"]:
                enqueue(lane_id, tid, chunk_id, t)

    max_steps = K * 20  # safety limit
    t = 0
    delivered = set()
    delivered_order = []

    while len(delivered) < total_chunks and t < max_steps:
        introduce_chunks(t)

        # Transmit, respecting shared physical-edge capacity
        forwarding = []
        get_phys = lambda tid, lane_id: tree_by_id[tid]["lane_phys"][lane_id]
        admitted = _admit_transmissions_credits(
            edge_queues, get_phys, r_phys, credits, bucket
        )
        for se, _vft, tid, chunk_id, lane_id in admitted:
            info = tree_by_id[tid]
            if emit is not None:
                # Pass precomputed physical edges; emit accumulates load per edge.
                emit(t, se, tid, chunk_id, info["lane_phys"][lane_id])
            else:
                schedule[t][se].append(
                    {
                        "tree_id": tid,
                        "chunk_id": chunk_id,
                        "metadata": info["lanes"][lane_id]["metadata"],
                    }
                )
            forwarding.append((tid, chunk_id, lane_id, t + 1))
            chunk_pending[(tid, chunk_id)] -= 1

        # Forward: after transmission on a lane, enqueue on successor lanes
        for tid, chunk_id, lane_id, next_t in forwarding:
            succs = tree_by_id[tid]["lane_successors"].get(lane_id, [])
            for next_lane_id in succs:
                enqueue(next_lane_id, tid, chunk_id, next_t)

        # Check for completed chunks
        for tid, chunk_id, lane_id, next_t in forwarding:
            if chunk_pending[(tid, chunk_id)] == 0 and (tid, chunk_id) not in delivered:
                delivered.add((tid, chunk_id))
                delivered_order.append((tid, chunk_id, t))
        t += 1

    if len(delivered) < total_chunks:
        logger.warning(
            "WFQ hit its safety limit (%d steps) with %d/%d chunks undelivered; "
            "the schedule is truncated and reported latencies are unreliable",
            max_steps,
            total_chunks - len(delivered),
            total_chunks,
        )

    raw_schedule = dict(schedule)

    if required_chunks is not None and required_chunks < total_chunks:
        raw_schedule = trim_schedule(raw_schedule, delivered_order, required_chunks)
    return raw_schedule


def wfq_allreduce_schedule(rs_trees, ag_trees, demand, C, K, R, bucket_extra=1.0):
    logger.info("WFQ AllReduce")
    if not rs_trees or not ag_trees:
        return {}, {}

    def _build_infos(trees):
        infos = []
        for tree_id, (edge_list, volume, _) in enumerate(trees):
            x_tau = volume * demand
            n_chunks = math.ceil(x_tau / C)
            if n_chunks == 0:
                continue
            root, root_lanes, lane_successors, spatial_edges, lanes, leaves = (
                build_tree_dag(edge_list)
            )
            rate = x_tau / (K * C)
            lane_phys = {
                lane_id: physical_edges(
                    lane["spatial"], lane["metadata"].get("switches", [])
                )
                for lane_id, lane in lanes.items()
            }
            infos.append(
                {
                    "id": tree_id,
                    "x_tau": x_tau,
                    "n_chunks": n_chunks,
                    "root_lanes": root_lanes,
                    "lane_successors": lane_successors,
                    "spatial_edges": spatial_edges,
                    "lanes": lanes,
                    "lane_phys": lane_phys,
                    "rate": rate,
                }
            )
        return infos

    rs_tree_infos = _build_infos(rs_trees)
    ag_tree_infos = _build_infos(ag_trees)

    if not rs_tree_infos or not ag_tree_infos:
        return {}, {}

    rs_by_id = {info["id"]: info for info in rs_tree_infos}
    ag_by_id = {info["id"]: info for info in ag_tree_infos}

    def _phys_rates(infos):
        r_phys = defaultdict(float)
        for info in infos:
            for pes in info["lane_phys"].values():
                for pe in set(pes):
                    r_phys[pe] += info["rate"]
        return r_phys

    r_phys_rs = _phys_rates(rs_tree_infos)
    r_phys_ag = _phys_rates(ag_tree_infos)
    cap_phys_rs = {pe: math.ceil(r) for pe, r in r_phys_rs.items()}
    cap_phys_ag = {pe: math.ceil(r) for pe, r in r_phys_ag.items()}
    bucket_rs = {pe: r + bucket_extra for pe, r in r_phys_rs.items()}
    bucket_ag = {pe: r + bucket_extra for pe, r in r_phys_ag.items()}
    credits_rs = defaultdict(float, dict(bucket_rs))
    credits_ag = defaultdict(float, dict(bucket_ag))

    def _intro_order(infos):
        pending = defaultdict(list)
        ordered = []
        for info in infos:
            tid = info["id"]
            rate = info["rate"]
            n = info["n_chunks"]
            chunks_so_far = 0
            t = 0
            while chunks_so_far < n:
                new_chunks = min(
                    math.ceil(rate * (t + 1)) - math.ceil(rate * t),
                    n - chunks_so_far,
                )
                for _ in range(new_chunks):
                    pending[t].append((tid, chunks_so_far))
                    ordered.append((tid, chunks_so_far))
                    chunks_so_far += 1
                t += 1
        return pending, ordered

    rs_pending_intro, rs_intro_order = _intro_order(rs_tree_infos)
    _ag_pending_ignored, ag_intro_order = _intro_order(ag_tree_infos)

    rs_global_id = {key: n for n, key in enumerate(rs_intro_order)}
    ag_global_id = {key: n for n, key in enumerate(ag_intro_order)}

    total_rs_chunks = len(rs_intro_order)
    total_ag_chunks = len(ag_intro_order)

    # WFQ state (separate for RS and AG)
    rs_queues = defaultdict(list)
    rs_v_prev = defaultdict(float)
    rs_chunk_pending = defaultdict(int)
    rs_chunk_enqueued = defaultdict(set)

    ag_queues = defaultdict(list)
    ag_v_prev = defaultdict(float)
    ag_chunk_pending = defaultdict(int)
    ag_chunk_enqueued = defaultdict(set)

    def _enqueue(
        queues,
        v_prev,
        chunk_pending,
        chunk_enqueued,
        by_id,
        r_phys,
        cap_phys,
        lane_id,
        tree_id,
        chunk_id,
        arrival_step,
    ):
        if lane_id in chunk_enqueued[(tree_id, chunk_id)]:
            return
        chunk_enqueued[(tree_id, chunk_id)].add(lane_id)
        info = by_id[tree_id]
        lane = info["lanes"][lane_id]
        se = lane["spatial"]
        x_tau = info["x_tau"]
        pes = info["lane_phys"][lane_id]
        bottleneck = max(r_phys[pe] / cap_phys[pe] for pe in pes)
        prev_vft = v_prev.get((tree_id, lane_id), 0)
        vft = max(arrival_step, prev_vft) + K * C * bottleneck / x_tau
        v_prev[(tree_id, lane_id)] = vft
        heapq.heappush(queues[se], (vft, tree_id, chunk_id, lane_id))
        chunk_pending[(tree_id, chunk_id)] += 1

    def rs_enqueue(lane_id, tree_id, chunk_id, arrival_step):
        _enqueue(
            rs_queues,
            rs_v_prev,
            rs_chunk_pending,
            rs_chunk_enqueued,
            rs_by_id,
            r_phys_rs,
            cap_phys_rs,
            lane_id,
            tree_id,
            chunk_id,
            arrival_step,
        )

    def ag_enqueue(lane_id, tree_id, chunk_id, arrival_step):
        _enqueue(
            ag_queues,
            ag_v_prev,
            ag_chunk_pending,
            ag_chunk_enqueued,
            ag_by_id,
            r_phys_ag,
            cap_phys_ag,
            lane_id,
            tree_id,
            chunk_id,
            arrival_step,
        )

    rs_schedule = defaultdict(lambda: defaultdict(list))
    ag_schedule = defaultdict(lambda: defaultdict(list))

    rs_delivered = set()
    ag_delivered = set()

    ag_intro_pending = defaultdict(list)  # step -> [(tree_id, chunk_id)]
    ag_next_to_introduce = 0

    max_steps = K * 40
    t = 0

    while len(ag_delivered) < total_ag_chunks and t < max_steps:
        # Introduce RS chunks (rate-based pacing)
        for tid, cid in rs_pending_intro.get(t, []):
            for lid in rs_by_id[tid]["root_lanes"]:
                rs_enqueue(lid, tid, cid, t)

        # Introduce AG chunks triggered by prior RS completions
        for tid, cid in ag_intro_pending.get(t, []):
            for lid in ag_by_id[tid]["root_lanes"]:
                ag_enqueue(lid, tid, cid, t)

        # Transmit RS, respecting shared physical-edge capacity
        rs_forwarding = []
        rs_admitted = _admit_transmissions_credits(
            rs_queues,
            lambda tid, lid: rs_by_id[tid]["lane_phys"][lid],
            r_phys_rs,
            credits_rs,
            bucket_rs,
        )
        for se, _vft, tid, cid, lid in rs_admitted:
            meta = rs_by_id[tid]["lanes"][lid]["metadata"]
            rs_schedule[t][se].append(
                {
                    "tree_id": tid,
                    "chunk_id": cid,
                    "global_id": rs_global_id[(tid, cid)],
                    "metadata": meta,
                }
            )
            rs_forwarding.append((tid, cid, lid, t + 1))
            rs_chunk_pending[(tid, cid)] -= 1

        for tid, cid, lid, next_t in rs_forwarding:
            for next_lid in rs_by_id[tid]["lane_successors"].get(lid, []):
                rs_enqueue(next_lid, tid, cid, next_t)

        # RS completion -> schedule next AG chunk for introduction
        for tid, cid, lid, next_t in rs_forwarding:
            if rs_chunk_pending[(tid, cid)] == 0 and (tid, cid) not in rs_delivered:
                rs_delivered.add((tid, cid))
                if ag_next_to_introduce < total_ag_chunks:
                    ag_tid, ag_cid = ag_intro_order[ag_next_to_introduce]
                    ag_next_to_introduce += 1
                    ag_intro_pending[t + 1].append((ag_tid, ag_cid))

        # Transmit AG, respecting shared physical-edge capacity
        ag_forwarding = []
        ag_admitted = _admit_transmissions_credits(
            ag_queues,
            lambda tid, lid: ag_by_id[tid]["lane_phys"][lid],
            r_phys_ag,
            credits_ag,
            bucket_ag,
        )
        for se, _vft, tid, cid, lid in ag_admitted:
            meta = ag_by_id[tid]["lanes"][lid]["metadata"]
            ag_schedule[t][se].append(
                {
                    "tree_id": tid,
                    "chunk_id": cid,
                    "global_id": ag_global_id[(tid, cid)],
                    "metadata": meta,
                }
            )
            ag_forwarding.append((tid, cid, lid, t + 1))
            ag_chunk_pending[(tid, cid)] -= 1

        for tid, cid, lid, next_t in ag_forwarding:
            for next_lid in ag_by_id[tid]["lane_successors"].get(lid, []):
                ag_enqueue(next_lid, tid, cid, next_t)

        for tid, cid, lid, next_t in ag_forwarding:
            if ag_chunk_pending[(tid, cid)] == 0 and (tid, cid) not in ag_delivered:
                ag_delivered.add((tid, cid))

        t += 1

    if len(ag_delivered) < total_ag_chunks:
        logger.warning(
            "all_reduce WFQ hit its safety limit (%d steps) with %d/%d all-gather "
            "chunks undelivered; the schedule is truncated and reported latencies "
            "are unreliable",
            max_steps,
            total_ag_chunks - len(ag_delivered),
            total_ag_chunks,
        )

    return dict(rs_schedule), dict(ag_schedule)


def wfq_path_schedule(paths, demand, C, K, R, required_chunks=None, bucket_extra=1.0):
    logger.info("WFQ Path")

    path_infos = []
    for path_id, (edge_list, volume, dest) in enumerate(paths):
        x_tau = volume * demand
        n_chunks = math.ceil(x_tau / C)
        if n_chunks == 0:
            continue
        rate = x_tau / (K * C)

        lanes = {}
        for idx, (u, v, t, meta) in enumerate(edge_list):
            lanes[idx] = {
                "src": u,
                "dst": v,
                "time": t,
                "spatial": (u, v),
                "metadata": meta if meta else {},
            }

        lane_phys = {
            lane_id: physical_edges(
                lane["spatial"], lane["metadata"].get("switches", [])
            )
            for lane_id, lane in lanes.items()
        }

        path_infos.append(
            {
                "id": path_id,
                "x_tau": x_tau,
                "n_chunks": n_chunks,
                "dest": dest,
                "lanes": lanes,
                "lane_phys": lane_phys,
                "rate": rate,
            }
        )

    if not path_infos:
        return {}

    path_by_id = {info["id"]: info for info in path_infos}

    # Expand each lane's elided edge into the physical sub-edges it actually
    # crosses, and aggregate rate/capacity per physical edge.
    r_phys = defaultdict(float)
    for info in path_infos:
        for pes in info["lane_phys"].values():
            for pe in set(pes):
                r_phys[pe] += info["rate"]

    capacity_phys = {pe: math.ceil(r) for pe, r in r_phys.items()}
    bucket = {pe: r + bucket_extra for pe, r in r_phys.items()}
    credits = defaultdict(float, dict(bucket))

    # Rate-based chunk introduction schedule
    pending_intro = defaultdict(list)
    for info in path_infos:
        pid = info["id"]
        rate = info["rate"]
        n = info["n_chunks"]
        chunks_so_far = 0
        t = 0
        while chunks_so_far < n:
            new_chunks = min(
                math.ceil(rate * (t + 1)) - math.ceil(rate * t),
                n - chunks_so_far,
            )
            for _ in range(new_chunks):
                pending_intro[t].append((pid, chunks_so_far))
                chunks_so_far += 1
            t += 1

    total_chunks = sum(info["n_chunks"] for info in path_infos)
    schedule = defaultdict(lambda: defaultdict(list))

    edge_queues = defaultdict(
        list
    )  # spatial_edge -> heap of (vft, path_id, chunk_id, lane_id)
    v_prev = defaultdict(float)
    chunk_pending = defaultdict(int)
    chunk_enqueued = defaultdict(set)

    def enqueue(lane_id, path_id, chunk_id, arrival_step):
        if lane_id in chunk_enqueued[(path_id, chunk_id)]:
            return
        chunk_enqueued[(path_id, chunk_id)].add(lane_id)

        info = path_by_id[path_id]
        lane = info["lanes"][lane_id]
        se = lane["spatial"]
        x_tau = info["x_tau"]
        pes = info["lane_phys"][lane_id]
        bottleneck = max(r_phys[pe] / capacity_phys[pe] for pe in pes)

        prev_vft = v_prev.get((path_id, lane_id), 0)
        vft = max(arrival_step, prev_vft) + K * C * bottleneck / x_tau
        v_prev[(path_id, lane_id)] = vft

        heapq.heappush(edge_queues[se], (vft, path_id, chunk_id, lane_id))
        chunk_pending[(path_id, chunk_id)] += 1

    def introduce_chunks(t):
        for pid, chunk_id in pending_intro.get(t, []):
            if path_by_id[pid]["lanes"]:
                enqueue(0, pid, chunk_id, t)

    max_steps = K * 20
    t = 0
    delivered = set()
    delivered_order = []

    while len(delivered) < total_chunks and t < max_steps:
        introduce_chunks(t)

        forwarding = []
        admitted = _admit_transmissions_credits(
            edge_queues,
            lambda pid, lane_id: path_by_id[pid]["lane_phys"][lane_id],
            r_phys,
            credits,
            bucket,
        )
        for se, _vft, pid, chunk_id, lane_id in admitted:
            info = path_by_id[pid]
            meta = info["lanes"][lane_id]["metadata"]
            schedule[t][se].append(
                {
                    "tree_id": pid,
                    "chunk_id": chunk_id,
                    "metadata": {**meta, "dest": info["dest"]},
                }
            )
            forwarding.append((pid, chunk_id, lane_id, t + 1))
            chunk_pending[(pid, chunk_id)] -= 1

        # Forward to successor lane (next edge along the path)
        for pid, chunk_id, lane_id, next_t in forwarding:
            next_lane_id = lane_id + 1
            if next_lane_id in path_by_id[pid]["lanes"]:
                enqueue(next_lane_id, pid, chunk_id, next_t)

        for pid, chunk_id, lane_id, next_t in forwarding:
            if chunk_pending[(pid, chunk_id)] == 0 and (pid, chunk_id) not in delivered:
                delivered.add((pid, chunk_id))
                delivered_order.append((pid, chunk_id, t))

        t += 1

    if len(delivered) < total_chunks:
        logger.warning(
            "path WFQ hit its safety limit (%d steps) with %d/%d chunks "
            "undelivered; the schedule is truncated and reported latencies are "
            "unreliable",
            max_steps,
            total_chunks - len(delivered),
            total_chunks,
        )

    raw_schedule = dict(schedule)

    if required_chunks is not None and required_chunks < total_chunks:
        raw_schedule = trim_schedule(raw_schedule, delivered_order, required_chunks)

    return raw_schedule
