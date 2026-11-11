// reach - the reachable sets of x' = A x (+ B u), as CORA's linearSys.reach
//
// input_set is a zonotope of u(t) in R^m; none means no input. alg::Standard has the options
// time_step, taylor_terms and zonotope_order; alg::WrappingFree time_step and taylor_terms;
// alg::FromStart those of Standard; alg::Decomposition those and the partition of the state.
// alg::Krylov (CORA's linAlg 'krylov') returns the output sets C x, as CORA's reach does, since
// it never forms a set of the state; a sparse A is used through its products only.
// time_point[k] is the set at k*time_step.
//
// Syntax:   R = sys.reach(problem, alg);
//           R = sys.reach({.initial_set = X0, .t_final = 2}, alg::Standard{.time_step = 0.1});
// Inputs:   problem - initial_set (zonotope), input_set (zonotope of u(t), needs B), t_final
//           alg - alg::Standard, alg::WrappingFree, alg::FromStart, alg::Decomposition, alg::Krylov
// Outputs:  R - time_interval[k] encloses the states over [k*time_step, (k+1)*time_step]
// See also: output_set, simulate, correction_matrix_state, private/priv_reach_standard
//
// CORA: contDynamics/@linearSys/reach.m
// CORA: contDynamics/@linearSys/private/priv_reach_standard.m
// CORA: contDynamics/@linearSys/private/priv_reach_wrappingfree.m
// CORA: contDynamics/@linearSys/private/priv_reach_fromStart.m
// CORA: contDynamics/@linearSys/private/priv_reach_decomp.m
// CORA: contDynamics/@linearSys/private/priv_reach_krylov.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/linear_sys/private/priv_krylov.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The steps ---------------------------------------------------------------------------------------

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
    // from_start is decomposition with the whole state as one block
    if (const auto *a = std::get_if<alg::FromStart>(&alg))
        return priv_reach_from_start(sys, X0, problem.input_set, a->time_step, a->taylor_terms,
                                     a->zonotope_order, priv_one_block(sys.dim()),
                                     priv_num_steps(t_final, a->time_step));
    if (const auto *a = std::get_if<alg::Decomposition>(&alg))
        return priv_reach_from_start(sys, X0, problem.input_set, a->time_step, a->taylor_terms,
                                     a->zonotope_order, a->partition,
                                     priv_num_steps(t_final, a->time_step));
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
        if (const auto *a = std::get_if<alg::Decomposition>(&alg))
            CORA_CHECK(a->partition.back()[1] == dim() - 1, Error::wrong_value, [&] {
                return "reach: the partition must end at the last state " +
                       std::to_string(dim() - 1) + ", got " +
                       std::to_string(a->partition.back()[1]);
            });
    }
    ReachSet R = aux_reach(*this, problem, alg);
    R.time_step = std::visit([](const auto &a) { return a.time_step; }, alg);
    return R;
}

// Krylov ------------------------------------------------------------------------------------------

ReachSet LinearSys::reach(const ReachProblem &problem, const alg::Krylov &alg) const {
    return reach(problem, alg, nullptr);
}

ReachSet LinearSys::reach(const ReachProblem &problem, const alg::Krylov &alg,
                          const std::function<void(const Zonotope &)> &on_time_interval) const {
    detail::Entry entry;
    // the problem and the options, then a single system and initial set
    if (entry.check()) {
        validate(problem);
        alg.validate();
    }
    CORA_CHECK(problem.initial_set->center().ndim() == 2 && (A_sparse_ || A_.ndim() == 2),
               Error::not_supported, [] {
                   return std::string("reach: alg::Krylov takes a single system and initial set");
               });
    ReachSet R = priv_reach_krylov(*this, *problem.initial_set, problem.input_set, alg,
                                   problem.t_final, on_time_interval);
    R.time_step = alg.time_step;
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
