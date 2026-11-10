// priv_reach_adaptive - the reachable sets of a nonlinear system with alg "lin-adaptive"
//
// Steps until the end time, each of a size the algorithm chooses (priv_adapt_step), from the set at
// the end of the step before. The first step starts at the initial set with the finite horizon,
// gain, error and order of the algorithm. The last step may end after t_final: its sets are those
// of a longer time span, which encloses the one asked for.
//
// Syntax:   ReachSet R = priv_reach_adaptive(sys, problem, alg);
// Inputs:   problem - the initial set (one zonotope), the input set, t_final
//           alg - the tuning constants
// Outputs:  R - time_interval[k] over [time[k], time[k+1]], time_point[k] at time[k]
// See also: priv_adapt_step, NonlinearSys::reach
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg) {
    const Zonotope &X0 = *problem.initial_set;
    // the state before the first step: no set has been reduced yet
    AdaptState state{
        0.0, X0, X0, alg.initial_step, alg.initial_gain, alg.initial_order, alg.initial_error};
    ReachSet R;
    R.time_point.push_back(X0);
    R.time.push_back(0.0);
    for (int k = 0; state.t < problem.t_final; ++k) {
        if (k >= alg.max_steps)
            CORA_FAIL(Error::reach_set_explosion,
                      "reach: the analysis did not reach t_final in max_steps steps (t = " +
                          std::to_string(state.t) + ")");
        state = priv_adapt_step(sys, problem.input_set, alg, state);
        R.time_interval.push_back(state.Rti);
        R.time_point.push_back(state.R);
        R.time.push_back(state.t);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
