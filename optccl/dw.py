import logging
import time
from collections import defaultdict

import gurobipy as gp
import numpy as np

from .errors import OptcclError, RmpInfeasibleError

logger = logging.getLogger(__name__)


def mirrored_dw(
    top,
    T,
    R,
    gurobi_env,
    formulation,
    *,
    name,
    no_crossover,
    phase1_alpha,
    dw_tolerance,
    feasible_only=False,
    rate_retry_max=0,
    rate_retry_factor=1.01,
):
    """Run the mirrored-DW column generation and return
    ``(m, w_arrays, f_arrays, g, elapsed, R)``.

    ``w_arrays`` / ``f_arrays`` map each EC representative to the convex
    combination of its extreme points (concatenated priced-group values /
    ``f_values`` layout respectively); origins that produced no columns are
    absent. The master may terminate merely feasible rather than optimal;
    a Phase 1 that converges with residual slack raises ``RmpInfeasibleError``.

    ``feasible_only`` returns the feasible Phase-1 solution the moment the RMP
    becomes feasible, skipping Phase 2 entirely -- the objective is ignored in
    favour of runtime.

    ``rate_retry_max`` (default 0 = disabled) enables rate-retry: when a solve
    raises ``RmpInfeasibleError`` (``R`` below the achievable rate), ``R`` is
    multiplied by ``rate_retry_factor`` and the solve is retried, up to
    ``rate_retry_max`` times. The trailing element of the returned tuple is the
    ``R`` actually used (== the input ``R`` unless retries bumped it), so callers
    report the effective rate.
    """
    start_time = time.time()

    def log(*args, level=logging.INFO):
        elapsed = time.time() - start_time
        msg = f"[{elapsed:8.2f}] " + " ".join(str(a) for a in args)
        logger.log(level, msg)

    attempts = 0
    while True:
        try:
            result = _mirrored_dw(
                top,
                T,
                R,
                gurobi_env,
                formulation,
                name,
                no_crossover,
                phase1_alpha,
                dw_tolerance,
                feasible_only,
                start_time,
                log,
            )
            return (*result, R)
        except RmpInfeasibleError:
            if rate_retry_max <= 0 or attempts >= rate_retry_max:
                raise
            attempts += 1
            new_R = R * rate_retry_factor
            log(
                f"{name}: RMP infeasible at R={R:.6g}; retry "
                f"{attempts}/{rate_retry_max} at R={new_R:.6g}",
                level=logging.WARNING,
            )
            R = new_R


def _mirrored_dw(
    top,
    T,
    R,
    gurobi_env,
    formulation,
    name,
    no_crossover,
    phase1_alpha,
    dw_tolerance,
    feasible_only,
    start_time,
    log,
):
    log("Beginning...")
    g = top.construct(T)
    gsize = g.size()
    m = gp.Model(env=gurobi_env)

    log("Generating graph and creating subproblems...")
    subproblems = {}
    for ec in top.ECs:
        sub_m = gp.Model(env=gurobi_env)
        sub_m.params.OutputFlag = 0
        if no_crossover:
            sub_m.params.Method = 2
            sub_m.params.Crossover = 0
            sub_m.params.BarHomogeneous = 1
            sub_m.params.ScaleFlag = 1
        subproblems[ec.gpus[0]] = sub_m

    edgelist = list(g.edges())

    log("Building formulation (variables and flow constraints)...")
    formulation.build(subproblems, g, edgelist)

    log("Generating linking constraints")
    dummy = m.addVar()
    constrs = []
    for bc in top.bandwidth_constraints:
        constrs.append(m.addConstr(dummy <= bc.bound * R, f"bandwidth {bc.name}"))

    slacks = []
    lambda_constrs = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        slack = m.addVar(ub=1, name=f"slack {o}")
        slacks.append(slack)
        lambda_constrs[o] = m.addLConstr(slack == 1, f"lambda {o}")

    # Two-phase column generation: Phase 1 minimizes the artificial slacks
    # alone (objective coeffs are O(1)) to reach a feasible RMP, then Phase 2
    # swaps in the real column costs.
    m.setObjective(sum(slacks), gp.GRB.MINIMIZE)
    phase = 1

    log("Computing coefficients...")
    A = np.zeros((len(constrs), gsize))
    etoi = {((i, j), t): idx for idx, ((i, t), (j, _)) in enumerate(edgelist)}

    # A layer-restricted edge (strengthening's per-(x,t) duplicates, via
    # top.edge_layers) only exists at some timesteps, so look up rather than
    # assume every (edge, t) is present.
    for ccount, bc in enumerate(top.bandwidth_constraints):
        for t in range(T):
            for edge in bc.edges:
                idx = etoi.get((edge, t))
                if idx is not None:
                    A[ccount, idx] = 1

    log("Symmetrizing matrix")
    A_syms = {}
    for ec in top.ECs:
        Asym = np.zeros((len(constrs), gsize))
        for shift in range(len(ec.gpus)):
            for i, ((ei, t1), (ej, _)) in enumerate(edgelist):
                new_idx = etoi[(ec.shift_fn(ei, shift), ec.shift_fn(ej, shift)), t1]
                Asym[:, i] += A[:, new_idx]
        A_syms[ec.gpus[0]] = Asym

    log("Objective coefficients")
    c_base = np.zeros(gsize)
    for idx, ((ei, _t), (ej, _)) in enumerate(edgelist):
        c_base[idx] = top.edge_data[(ei, ej)]

    log("Symmetrizing objective")
    c_syms = {}
    for ec in top.ECs:
        c_base_sym = np.zeros(gsize)
        for i, ((ei, t1), (ej, _)) in enumerate(edgelist):
            for shift in range(len(ec.gpus)):
                new_idx = etoi[(ec.shift_fn(ei, shift), ec.shift_fn(ej, shift)), t1]
                c_base_sym[i] += c_base[new_idx]
        c_syms[ec.gpus[0]] = c_base_sym

    alpha = phase1_alpha
    phase1_cost_weight = 0.0
    cost_scale = None

    extreme_points_f = []
    extreme_points_w = []
    extreme_points_idxs = defaultdict(list)
    column_costs = []  # true (Phase-2) cost of each extreme point

    pi = np.zeros(len(constrs))
    ts = {ec.gpus[0]: np.inf for ec in top.ECs}

    def switch_to_phase2():
        nonlocal phase
        for ec2 in top.ECs:
            oo = ec2.gpus[0]
            for idx, var in extreme_points_idxs[oo]:
                var.Obj = column_costs[idx]
        slack_penalty = 10.0 * max(max(column_costs, default=1.0), 1.0)
        for slk in slacks:
            slk.Obj = slack_penalty
        phase = 2

    def solve_subproblem(sub):
        sub.reset()
        sub.optimize()
        if phase == 1 and sub.Status == gp.GRB.NUMERIC:
            log(
                "  pricing solve status NUMERIC; raising numeric focus and retrying",
                level=logging.DEBUG,
            )
            sub.params.NumericFocus = 3
            sub.reset()
            sub.optimize()
            sub.params.NumericFocus = 0
            if sub.Status != gp.GRB.OPTIMAL:
                log(
                    f"pricing solve status {sub.Status} after raising numeric focus",
                    level=logging.WARNING,
                )

    def price_all():
        idxs = []
        for ec in top.ECs:
            o = ec.gpus[0]
            log(f" trying subproblem {o}", level=logging.DEBUG)
            cw = c_syms[o] if phase == 2 else phase1_cost_weight * c_syms[o]
            rc = cw - pi.T @ A_syms[o]
            obj = sum(
                rc[i] * v
                for group in formulation.priced_groups(o)
                for i, v in enumerate(group)
            )
            subproblems[o].setObjective(obj, gp.GRB.MINIMIZE)
            solve_subproblem(subproblems[o])
            if subproblems[o].SolCount == 0:
                # No usable solution (e.g. a persistent NUMERIC failure): treat as
                # "no improving column" so termination proceeds cleanly instead of
                # crashing on an unavailable objVal.
                log(
                    f"  subproblem {o} has no solution (status {subproblems[o].Status}); skipping",
                    level=logging.DEBUG,
                )
                continue
            sub_obj = subproblems[o].objVal
            if phase == 2:
                improving = (ts[o] - sub_obj) / max(abs(sub_obj), 1e-10) > dw_tolerance
            else:
                # sub_obj can be ~0 in Phase 1, so use an absolute reduced-cost test.
                improving = ts[o] - sub_obj > dw_tolerance
            if improving:
                log(
                    f"Improving by {ts[o]}, {sub_obj}, {ts[o] - sub_obj}",
                    level=logging.DEBUG,
                )
                idxs.append(o)
        return idxs

    iteration = 0
    slack_sum = len(slacks)
    stall_limit = 200
    best_progress = np.inf
    stall_count = 0
    while True:
        iteration += 1
        log(f"ITERATION {iteration} BEGIN", level=logging.DEBUG)

        if iteration == 1:
            m.addConstr(dummy <= 0)

        m.optimize()

        if phase == 1:
            slack_sum = sum(s.X for s in slacks)
            if slack_sum <= 1e-6:
                if feasible_only:
                    log(
                        f"Phase 1 reached feasibility (slack sum {slack_sum:.3e}) "
                        f"at iteration {iteration}; feasible_only set, returning "
                        f"this feasible solution without running Phase 2."
                    )
                    break
                log(
                    f"Phase 1 reached feasibility (slack sum {slack_sum:.3e}) at "
                    f"iteration {iteration}; switching to Phase 2."
                )
                switch_to_phase2()
                m.optimize()
                best_progress = np.inf
                stall_count = 0

        progress = slack_sum if phase == 1 else m.ObjVal
        if progress < best_progress - dw_tolerance * max(1.0, abs(best_progress)):
            best_progress = progress
            stall_count = 0
        else:
            stall_count += 1
        stalled = stall_count >= stall_limit
        if stalled:
            log(
                f"Master has not improved in {stall_limit} iterations "
                f"(phase {phase}); treating pricing as converged.",
                level=logging.WARNING,
            )

        for i in range(len(pi)):
            pi[i] = constrs[i].pi
        for ec in top.ECs:
            o = ec.gpus[0]
            ts[o] = lambda_constrs[o].pi

        improvement_indices = price_all()

        if not improvement_indices or stalled:
            if phase == 1:
                if phase1_cost_weight != 0.0:
                    log(
                        f"Phase 1 stalled with residual slack {slack_sum:.3e} "
                        f"under cost weight {phase1_cost_weight:.3e}; disabling the "
                        f"Phase-1 cost weight and continuing as pure feasibility."
                    )
                    for ec2 in top.ECs:
                        oo = ec2.gpus[0]
                        for idx, var in extreme_points_idxs[oo]:
                            var.Obj = 0.0
                    phase1_cost_weight = 0.0
                    best_progress = np.inf
                    stall_count = 0
                    continue
                # Pure-feasibility Phase 1 converged with slacks remaining => infeasible.
                log(
                    f"Phase 1 converged with residual slack {slack_sum:.3e}; "
                    f"RMP is infeasible (R below the achievable rate).",
                    level=logging.WARNING,
                )
                raise RmpInfeasibleError(
                    f"{name} RMP infeasible after Phase 1 "
                    f"(residual artificial slack {slack_sum:.3e}). R is below the "
                    f"collective's achievable rate on this topology -- the default "
                    f"capacity bound can be unachievable when copy/reduce "
                    f"capabilities differ (e.g. reduce_scatter with copy-only mem "
                    f"nodes). Pass a larger -r."
                )
            break

        for o in improvement_indices:
            gvals = [
                np.array([v.X for v in group]) for group in formulation.priced_groups(o)
            ]
            # A rep may carry no commodities of its own (e.g. it only appears as
            # a sink of another rep's demand); its column is the zero column.
            wval = np.concatenate(gvals) if gvals else np.zeros(0)
            gsum = np.sum(gvals, axis=0) if gvals else np.zeros(gsize)

            cost = float(c_syms[o] @ gsum)

            added_constrs = [lambda_constrs[o]]
            added_coeffs = [1]
            coeffs = A_syms[o] @ gsum
            for j, constr in enumerate(constrs):
                if coeffs[j] > 0:
                    added_constrs.append(constr)
                    added_coeffs.append(coeffs[j])

            col_obj = cost if phase == 2 else phase1_cost_weight * cost
            var = m.addVar(obj=col_obj, column=gp.Column(added_coeffs, added_constrs))
            extreme_points_w.append(wval)
            extreme_points_f.append(formulation.f_values(o))
            column_costs.append(cost)
            extreme_points_idxs[o].append((len(extreme_points_w) - 1, var))

        if phase == 1 and cost_scale is None and alpha != 0.0:
            # Calibrate the Phase-1 cost weight and apply it to the columns just
            # added (which entered with obj 0).
            g_all = np.concatenate(
                [np.abs(pi @ A_syms[ec2.gpus[0]]) for ec2 in top.ECs]
            )
            c_all = np.concatenate([np.abs(c_syms[ec2.gpus[0]]) for ec2 in top.ECs])
            g_scale = float(g_all[g_all > 0].mean()) if np.any(g_all > 0) else 0.0
            c_scale = float(c_all[c_all > 0].mean()) if np.any(c_all > 0) else 0.0
            if c_scale > 0 and g_scale > 0:
                cost_scale = c_scale
                phase1_cost_weight = alpha * g_scale / c_scale
                log(
                    f"Calibrated Phase-1 cost weight: g_scale={g_scale:.3e}, "
                    f"c_scale={c_scale:.3e}, weight={phase1_cost_weight:.3e}"
                )
                for ec2 in top.ECs:
                    oo = ec2.gpus[0]
                    for idx, var in extreme_points_idxs[oo]:
                        var.Obj = phase1_cost_weight * column_costs[idx]

    w_arrays = {}
    f_arrays = {}
    for ec in top.ECs:
        o = ec.gpus[0]
        if not extreme_points_idxs[o]:
            continue
        w_arrays[o] = sum(
            extreme_points_w[idx] * var.X for idx, var in extreme_points_idxs[o]
        )
        f_arrays[o] = sum(
            extreme_points_f[idx] * var.X for idx, var in extreme_points_idxs[o]
        )
    log("Done!")
    return m, w_arrays, f_arrays, g, time.time() - start_time
