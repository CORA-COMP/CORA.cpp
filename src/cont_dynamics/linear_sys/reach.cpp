// reach - the reachable sets of x' = A x (+ B u), as CORA's linearSys.reach
//
// Syntax:   R = sys.reach(problem, alg);
//           R = sys.reach({.initial_set = X0, .t_final = 2}, alg::Standard{.time_step = 0.1});
// Inputs:   problem - initial_set (zonotope), input_set (zonotope of u(t) in R^m, needs B; none:
//                     no input), t_final
//           alg - alg::Standard (time_step, taylor_terms, zonotope_order) or alg::WrappingFree
//                 (time_step, taylor_terms)
// Outputs:  R - time_interval[k] encloses the states over [k*time_step, (k+1)*time_step];
//           time_point[k] is the set at k*time_step
// See also: output_set, simulate, correction_matrix_state, private/priv_reach_standard
//
// CORA: contDynamics/@linearSys/reach.m
// CORA: contDynamics/@linearSys/private/priv_reach_standard.m
// CORA: contDynamics/@linearSys/private/priv_reach_wrappingfree.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The step's propagation e^{A dt}, its correction matrix F and the number of steps.
struct Step {
    Tensor e_adt;
    Interval F;
    int steps;
};

Step aux_step(const LinearSys &sys, double time_step, int taylor_terms, double t_final) {
    return {(sys.A() * time_step).expm(), sys.correction_matrix_state(time_step, taylor_terms),
            priv_num_steps(t_final, time_step)};
}

// The sets of the algorithm, without the step size that the result records.
ReachSet aux_reach(const LinearSys &sys, const ReachProblem &problem, const alg::LinearReach &alg) {
    const Zonotope &X0 = *problem.initial_set;
    const double t_final = problem.t_final;
    if (problem.input_set) {
        // The step's propagation, curvature and input solution are the same for every step.
        const Zonotope &U = *problem.input_set;
        if (const auto *a = std::get_if<alg::Standard>(&alg)) {
            const Step s = aux_step(sys, a->time_step, a->taylor_terms, t_final);
            const InputSolution in = sys.input_solution(U, a->time_step, a->taylor_terms);
            return priv_reach_standard(X0, s.e_adt, s.F, in, s.steps, a->zonotope_order);
        }
        const auto &a = std::get<alg::WrappingFree>(alg);
        const Step s = aux_step(sys, a.time_step, a.taylor_terms, t_final);
        const InputSolution in = sys.input_solution(U, a.time_step, a.taylor_terms);
        return priv_reach_wrappingfree(X0, s.e_adt, s.F, in, s.steps);
    }
    if (const auto *a = std::get_if<alg::Standard>(&alg)) {
        const Step s = aux_step(sys, a->time_step, a->taylor_terms, t_final);
        return priv_reach_standard(X0, s.e_adt, s.F, s.steps);
    }
    const auto &a = std::get<alg::WrappingFree>(alg);
    const Step s = aux_step(sys, a.time_step, a.taylor_terms, t_final);
    return priv_reach_wrappingfree(X0, s.e_adt, s.F, s.steps);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet LinearSys::reach(const ReachProblem &problem, const alg::LinearReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate(problem);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    ReachSet R = aux_reach(*this, problem, alg);
    R.time_step = std::visit([](const auto &a) { return a.time_step; }, alg);
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
