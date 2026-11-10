// priv_reach_adaptive - the reachable sets of a nonlinear system with alg "lin-adaptive"
//
// Steps until the end time, each of a size the algorithm chooses (priv_lin_reach_adaptive),
// from the set at the end of the step before. The set over a step is reduced adaptively at five
// times the zeta_z of the set at its end, which is reduced at zeta_z and starts the next step.
// The analysis stops early, with a warning, if the step size shrinks so that the rest of the time
// span would take more than 10^9 times the last ten steps.
//
// Syntax:   ReachSet R = priv_reach_adaptive(sys, problem, alg);
// Inputs:   problem - the initial set (one zonotope), the input set, t_final
//           alg - the tuning constants
// Outputs:  R - time_interval[k] over [time[k], time[k+1]], time_point[k] at time[k]
// See also: priv_lin_reach_adaptive, NonlinearSys::reach
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA: contDynamics/@nonlinearSys/initReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the step sizes have converged to 0 (CORA's aux_checkForAbortion).
bool aux_abort(const std::vector<double> &steps, double t, double t_final) {
    const std::size_t last = std::min<std::size_t>(10, steps.size());
    double sum = 0;
    for (std::size_t k = steps.size() - last; k < steps.size(); ++k) sum += steps[k];
    if ((t_final - t) / sum <= 1e9) return false;
    warn("CORA:contDynamics", "The analysis is aborted because the time step size converges to 0. "
                              "The reachable sets until t = " +
                                  std::to_string(t) + " are returned.");
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg) {
    const Zonotope &X0 = *problem.initial_set;
    // the options of the analysis, from the algorithm and the problem
    AdaptiveRun run;
    run.alg = alg;
    run.t_final = problem.t_final;
    run.U = problem.input_set;
    if (const int64_t m = sys.input_dim(); m > 0)
        run.U = problem.input_set
                    ? *problem.input_set
                    : Zonotope(X0.center().zeros_like({m, 1}), X0.center().zeros_like({m, 0}));
    if (run.U) run.U0 = *run.U - run.U->center();
    run.error_adm_horizon = X0.center().zeros_like();
    run.error_adm_delta_t_opt = X0.center().zeros_like();

    // the sets and the step sizes so far
    ReachSet R;
    R.time_point.push_back(X0);
    R.time.push_back(0.0);
    std::vector<double> steps;
    Zonotope current = X0;
    bool abort = false;
    // the steps, each from the set at the end of the one before
    while (run.t_final - run.t > numerics::kHorizonEnd && !abort) {
        const AdaptiveStep step = priv_lin_reach_adaptive(sys, current, run);
        R.time_interval.push_back(step.time_interval.reduce_adaptive(5 * alg.zeta_z).Z);
        current = step.time_point.reduce_adaptive(alg.zeta_z).Z;
        R.time_point.push_back(current);
        steps.push_back(run.time_step);
        run.t = run.t + run.time_step;
        R.time.push_back(run.t);
        ++run.i;
        abort = aux_abort(steps, run.t, run.t_final);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
