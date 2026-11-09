// reach - the reachable sets of x' = f(x, u), as CORA's nonlinearSys.reach with algorithm "lin"
//
// The first step is init_reach from the initial set, every further one post from the end of the
// step before: each linearizes f at the center of the set moved half a step along the flow, solves
// the affine system with the input set U, and adds the solution of the linearization error
// (CORA's linReach, private/priv_lin_reach.cpp). post reduces both sets to zonotope_order; the
// first step is not reduced, as in CORA. The error of a step is the first guess of the next.
//
// Syntax:   R = sys.reach(problem, alg);
//           R = sys.reach({.initial_set = X0, .t_final = 1}, alg::Linearization{.time_step = .01});
// Inputs:   problem - initial_set (one zonotope), input_set (of the m inputs, optional), t_final
//           alg - alg::Linearization (time_step, taylor_terms, zonotope_order and the orders)
// Outputs:  R - time_interval[k] encloses the states over [k*time_step, (k+1)*time_step]
// See also: linearize, simulate, LinearSys::reach
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinearSys/initReach.m
// CORA: contDynamics/@nonlinearSys/post.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The checks --------------------------------------------------------------------------------------

void NonlinearSys::validate_sets(const char *op, const Zonotope &R,
                                 const std::optional<Zonotope> &U) const {
    detail::Validation entry;
    CORA_CHECK(R.center().shape() == std::vector<int64_t>({n_, 1}) &&
                   R.generators().shape().size() == 2,
               Error::not_supported, [&] {
                   return std::string(op) + ": the set of a nonlinear system is one zonotope of " +
                          "the dimension " + std::to_string(n_) + "; batches are not supported";
               });
    CORA_CHECK(!U || m_ > 0, Error::not_supported, [&] {
        return std::string(op) + ": the system has no inputs; leave the input set empty";
    });
    CORA_CHECK(!U || (U->center().shape() == std::vector<int64_t>({m_, 1}) &&
                      U->generators().shape().size() == 2),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the input set must be one zonotope of the " +
                          std::to_string(m_) + " inputs";
               });
}

// The steps ---------------------------------------------------------------------------------------

NonlinearStep NonlinearSys::init_reach(const Zonotope &R0, const std::optional<Zonotope> &U,
                                       const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("init_reach", R0, U);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    // CORA starts from the error 0
    const alg::Linearization &options = std::get<alg::Linearization>(alg);
    return priv_lin_reach(*this, R0, R0.center().zeros_like(), U, options);
}

// CORA's post reduces the sets that initReach returns, and keeps the error.
NonlinearStep NonlinearSys::post(const NonlinearStep &step, const std::optional<Zonotope> &U,
                                 const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("post", step.time_point, U);
        std::visit([](const auto &a) { a.validate(); }, alg);
        CORA_CHECK(step.error.defined() && step.error.shape() == std::vector<int64_t>({n_, 1}),
                   Error::dimension_mismatch, [&] {
                       return "post: the error of the step must be a column (" +
                              std::to_string(n_) + ", 1)";
                   });
    }
    const alg::Linearization &options = std::get<alg::Linearization>(alg);
    NonlinearStep next = priv_lin_reach(*this, step.time_point, step.error, U, options);
    next.time_point = next.time_point.reduce(options.zonotope_order);
    next.time_interval = next.time_interval.reduce(options.zonotope_order);
    return next;
}

// The reachable sets ------------------------------------------------------------------------------

ReachSet NonlinearSys::reach(const ReachProblem &problem, const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    // one initial zonotope of the dimension, an input set of the inputs, a positive horizon
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) {
        validate_sets("reach", *problem.initial_set, problem.input_set);
        std::visit([](const auto &a) { a.validate(); }, alg);
    }
    // the start set, the first step, then the step after each one
    const double time_step = std::get<alg::Linearization>(alg).time_step;
    const int steps = priv_num_steps(problem.t_final, time_step);
    ReachSet R;
    R.time_point.push_back(*problem.initial_set);
    NonlinearStep step = init_reach(*problem.initial_set, problem.input_set, alg);
    for (int k = 1; k < steps; ++k) {
        R.time_interval.push_back(step.time_interval);
        R.time_point.push_back(step.time_point);
        step = post(step, problem.input_set, alg);
    }
    R.time_interval.push_back(step.time_interval);
    R.time_point.push_back(step.time_point);
    R.time_step = time_step;
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
