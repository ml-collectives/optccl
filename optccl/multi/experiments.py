from ..errors import OptcclError
from .workload import max_demand, subworkload
from .solver import multiple_collective_generic_mirrored_dw, determine_throughput
from .process import (
    FAMILY_ORDER,
    PROCESS_FAMILY,
    process_p2p,
    residual_rhs,
    merge_step_aligned,
    summarize,
)

_KB_PER_GB = 1e6


def _demand_arg(cfg, wl):
    return cfg.total_data_size / max_demand(wl)


def _seconds(raw):
    return None if raw is None else raw / _KB_PER_GB


def _report(
    mode,
    workload_name,
    *,
    scale,
    ideal,
    realized,
    components,
    breakdown,
    solve_time,
    analyses=None,
):
    return {
        "mode": mode,
        "workload": workload_name,
        "scale": scale,
        "ideal_latency": ideal,
        "realized_latency": realized,
        "components": components,
        "breakdown": breakdown,
        "solve_time": solve_time,
        "_analyses": analyses or {},
    }


def _solve(wl, T, R, gurobi_env, cfg):
    m, f_sol, w_sol, _edge_util, elapsed, R = multiple_collective_generic_mirrored_dw(
        wl,
        T,
        R,
        gurobi_env,
        no_crossover=True,
        phase1_cost_weight_alpha=cfg.multi_phase1_cost_weight,
        dw_tolerance=cfg.dw_tolerance,
        feasible_only=cfg.stop_at_feasible,
        rate_retry_max=cfg.rate_retry_max,
        rate_retry_factor=cfg.rate_retry_factor,
    )
    # R may have been bumped by rate-retry; return the effective rate so the
    # caller decomposes against the matching budget and reports the right ideal.
    return f_sol, w_sol, elapsed, R


def _families(wl):
    present = {"p2p": bool(wl.p2ps), "g": bool(wl.gathers), "r": bool(wl.reduces)}
    return [fam for fam in FAMILY_ORDER if present[fam]]


def _decompose_families(wl, f_sol, w_sol, demand_arg, R, gurobi_env, cfg, T):
    v = wl.topology_view()
    C, K = cfg.uniform_chunk_size, cfg.step_schedule_K
    g = v.construct(T)

    fams = _families(wl)
    full_budget = len(fams) == 1

    results = {}
    prior = []
    for fam in fams:
        if fam == "p2p":
            results[fam] = process_p2p(v, f_sol, demand_arg, C, K, R, cfg, T)
        else:
            override = None
            if prior and len(v.ECs) == 1:
                override = residual_rhs(v, g, w_sol, R, prior)
            results[fam] = PROCESS_FAMILY[fam](
                v,
                f_sol,
                w_sol,
                demand_arg,
                C,
                K,
                R,
                gurobi_env,
                cfg,
                T,
                bc_rhs_override=override,
                full_budget=full_budget,
            )
        prior.append(fam)
    return results


_FAMILY_LABELS = {"p2p": "p2p", "g": "gather", "r": "reduce"}


def process_joint(wl, gurobi_env, cfg, R=None):
    da = _demand_arg(cfg, wl)
    s = da / _KB_PER_GB
    T = cfg.time_steps
    Rj = R if R is not None else determine_throughput(wl, gurobi_env)
    f_sol, w_sol, t_solve, Rj = _solve(wl, T, Rj, gurobi_env, cfg)

    fam_loads = _decompose_families(wl, f_sol, w_sol, da, Rj, gurobi_env, cfg, T)

    merged = merge_step_aligned([load for load, _ns in fam_loads.values()])
    comb_scal, comb_an = summarize(merged, max(ns for _load, ns in fam_loads.values()))
    breakdown = {}
    for fam, (load, ns) in fam_loads.items():
        real = summarize(load, ns)[0]["realized_latency"]
        breakdown[_FAMILY_LABELS[fam]] = {"ideal": None, "realized": _seconds(real)}
    return _report(
        "joint",
        wl.name,
        scale=s,
        ideal=Rj * s,
        realized=_seconds(comb_scal["realized_latency"]),
        components={"R_combined": Rj},
        breakdown=breakdown,
        solve_time=t_solve,
        analyses={"combined": comb_an},
    )


def _grouped(wl, key_attr):
    groups = {}
    for ce in wl.collectives:
        groups.setdefault(getattr(ce, key_attr), []).append(ce)
    return groups


def _solve_group(wl, gname, entries, gurobi_env, cfg, demand_arg, T):
    sub = subworkload(wl, entries, name=f"{wl.name}:{gname}")
    R_g = determine_throughput(sub, gurobi_env)
    f_sol, w_sol, t_solve, R_g = _solve(sub, T, R_g, gurobi_env, cfg)
    fam_loads = _decompose_families(
        sub, f_sol, w_sol, demand_arg, R_g, gurobi_env, cfg, T
    )
    merged = merge_step_aligned([load for load, _ns in fam_loads.values()])
    scal, _an = summarize(merged, max(ns for _load, ns in fam_loads.values()))
    return R_g, scal["realized_latency"], t_solve


def _process_additive(mode, key_attr, wl, gurobi_env, cfg):
    da = _demand_arg(cfg, wl)
    s = da / _KB_PER_GB
    T = cfg.time_steps

    ideal = 0.0
    realized = 0.0
    total_solve = 0.0
    components = {}
    breakdown = {}
    for gname, entries in _grouped(wl, key_attr).items():
        repeats = 1
        if mode == "concat":
            rset = {ce.sequential_repeats for ce in entries}
            if len(rset) != 1:
                raise OptcclError(
                    f"concat group {gname!r}: entries disagree on "
                    f"sequential_repeats ({sorted(rset)})"
                )
            repeats = rset.pop()
        R_g, real_raw, t_solve = _solve_group(
            wl, gname, entries, gurobi_env, cfg, da, T
        )
        ideal += repeats * R_g * s
        realized += repeats * _seconds(real_raw)
        total_solve += t_solve
        components[f"R_{gname}"] = R_g
        if repeats != 1:
            components[f"repeats_{gname}"] = repeats
        breakdown[gname] = {
            "ideal": repeats * R_g * s,
            "realized": repeats * _seconds(real_raw),
        }
    return _report(
        mode,
        wl.name,
        scale=s,
        ideal=ideal,
        realized=realized,
        components=components,
        breakdown=breakdown,
        solve_time=total_solve,
    )


def process_overlay(wl, gurobi_env, cfg, R=None):
    return _process_additive("overlay", "group", wl, gurobi_env, cfg)


def process_concat(wl, gurobi_env, cfg, R=None):
    return _process_additive("concat", "concat_group", wl, gurobi_env, cfg)


PROCESSORS = {
    "concat": process_concat,
    "overlay": process_overlay,
    "joint": process_joint,
}


def run_mode(mode, wl, gurobi_env, cfg, R=None):
    return PROCESSORS[mode](wl, gurobi_env, cfg, R=R)
