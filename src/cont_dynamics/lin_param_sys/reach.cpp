// reach - the reachable sets of a linear parametric system, as CORA's reach of a linParamSys
//
// As CORA's validateOptions, the input set loses its center, which becomes the constant input
// u_trans (no input set: the origin). The steps cover [0, t_final] as MATLAB's 0:dt:t_final:
// init_reach gives the first, post every further one. CORA's check of a set explosion is not
// ported.
//
// Syntax:   ReachSet R = sys.reach(problem, alg::LinParamStandard{.time_step = 0.01});
// Inputs:   problem - initial_set (n), input_set (m, needs B; optional), t_final > 0
//           alg - alg::LinParamStandard
// Outputs:  R - time_interval[k] over [k dt, (k+1) dt], time_point[k] at k dt (or none)
// See also: init_reach, post, LinearSys::reach
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ReachSet LinParamSys::reach(const ReachProblem &problem, const alg::LinParamReach &alg) const {
    detail::Entry entry;
    const auto &a = std::get<alg::LinParamStandard>(alg);
    if (entry.check()) a.validate();
    // an initial set, a positive horizon of at least one step, an input set of B's columns
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(std::isfinite(problem.t_final) && problem.t_final >= a.time_step, Error::wrong_value,
               [&] {
                   return "reach: t_final must be finite and at least one time_step, got " +
                          std::to_string(problem.t_final);
               });
    CORA_CHECK(!problem.input_set || (B_ && problem.input_set->dim() == input_dim()),
               Error::dimension_mismatch, [&] {
                   return "reach: the input set needs B and its " + std::to_string(input_dim()) +
                          " columns, got the dimension " + std::to_string(problem.input_set->dim());
               });
    const Zonotope &R0 = *problem.initial_set;
    // the input set without its center, which is the constant input (CORA's uTrans)
    const int64_t m = B_ ? input_dim() : 1;
    Zonotope U(R0.center().zeros_like({m, 1}), R0.center().zeros_like({m, 0}));
    Tensor u_trans = U.center();
    if (problem.input_set) {
        U = *problem.input_set;
        bool shifted = false;
        for (double v : U.center().data()) shifted = shifted || v != 0;
        if (shifted) {
            u_trans = U.center();
            U = U - u_trans;
        }
    }
    // the steps of 0:dt:t_final
    const auto steps =
        static_cast<int>(std::floor(problem.t_final / a.time_step + numerics::kStepCount));
    ReachSet out;
    out.time_step = a.time_step;
    if (a.comp_time_point) out.time_point.push_back(R0);
    LinParamStep R = init_reach(R0, U, u_trans, a);
    for (int k = 0; k < steps; ++k) {
        // the sets of step k, then the next step
        if (k > 0) R = priv_next_step(R, a);
        out.time_interval.push_back(R.time_interval);
        if (R.time_point) out.time_point.push_back(*R.time_point);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
