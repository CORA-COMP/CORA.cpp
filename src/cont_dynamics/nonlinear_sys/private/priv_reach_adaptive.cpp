// priv_reach_adaptive - the reachable sets of a nonlinear system with alg "lin-adaptive"
//
// Steps until the end time, each of a size the algorithm chooses (priv_lin_reach_adaptive), from
// the set at the end of the step before. Both sets of a step are reduced adaptively, the one at its
// end at zeta_z and the one over it at five times zeta_z. The analysis stops, with a warning, if
// the step size shrinks so that the rest of the time span would take more than 10^9 times the last
// ten steps (CORALean's reachAdaptiveFloat, CORA's reach_adaptive).
//
// Syntax:   ReachSet R = priv_reach_adaptive(sys, problem, alg);
// Inputs:   problem - the initial set (one zonotope), the input set, t_final
//           alg - the tuning constants
// Outputs:  R - time_interval[k] over [time[k], time[k+1]], time_point[k] at time[k]
// See also: priv_lin_reach_adaptive, NonlinearSys::reach
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the step sizes have converged to 0 (CORA's aux_checkForAbortion): the remaining time
/// is more than 10^9 times the last ten steps.
bool aux_abort(const Vec &steps, double t, double t_final) {
    const std::size_t last = std::min<std::size_t>(10, steps.size());
    double sum = 0;
    for (std::size_t k = 0; k < last; ++k) sum = sum + steps[k];
    return !steps.empty() && 1e9 * sum < t_final - t;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg) {
    const Zonotope &X0 = *problem.initial_set;
    const int64_t n = sys.dim(), m = sys.input_dim();
    std::optional<Zonotope> U = problem.input_set;
    if (m > 0 && !U) U = Zonotope(X0.center().zeros_like({m, 1}), X0.center().zeros_like({m, 0}));

    // the state before the first step (linReachAdaptiveInit)
    AdaptiveState st(X0, n, 0.0);
    ReachSet R;
    R.time_point.push_back(X0);
    R.time.push_back(0.0);

    // the steps, each reduced for the next
    while (numerics::kHorizonEnd < problem.t_final - st.t &&
           !aux_abort(st.steps, st.t, problem.t_final)) {
        st = priv_lin_reach_adaptive(sys, U, alg, problem.t_final, st);
        st.R = st.R.reduce_adaptive(alg.zeta_z);
        st.Rti = st.Rti.reduce_adaptive(alg.zeta_z * 5);
        R.time_interval.push_back(st.Rti);
        R.time_point.push_back(st.R);
        R.time.push_back(st.t);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
