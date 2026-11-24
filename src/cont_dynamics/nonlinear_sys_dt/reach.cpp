// reach - the reachable sets of x(k+1) = f(x(k), u(k)), as CORA's nonlinearSysDT.reach
//
// Each step linearizes f at the center of the set (post, CORA's linReach), and the output set of
// each state set is computed (output_set), unless the algorithm skips it. The sets are those at
// the multiples of the sampling time. A specification and a constant input uTrans (uTransVec) are
// not supported; a tracked input trajectory is the input set of the problem.
//
// Syntax:   R = sys.reach(problem, alg);   R = sys.reach({.initial_set = X0, .t_final = 1}, alg);
// Inputs:   problem - initial_set (one zonotope), input_set (of the m inputs, optional), t_final
//           alg - alg::DiscreteLinearization (zonotope_order, the tensor orders)
// Outputs:  R - the states[k], outputs[k] at k*time_step and the linearization errors
// See also: post, output_set, simulate
//
// CORA: contDynamics/@nonlinearSysDT/reach.m
// CORA: contDynamics/@nonlinearSysDT/post.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys_dt/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

void NonlinearSysDT::validate_sets(const char *op, const Zonotope &R,
                                   const std::optional<Zonotope> &U) const {
    detail::Validation entry;
    // one zonotope of the states and, if any, one of the inputs
    CORA_CHECK(R.center().shape() == std::vector<int64_t>({dim()}), Error::not_supported, [&] {
        return std::string(op) +
               ": the set of a nonlinear system is one zonotope of the dimension " +
               std::to_string(dim()) + "; batches are not supported";
    });
    // the input set exists only for a system with inputs, and has their number
    CORA_CHECK(!U || input_dim() > 0, Error::not_supported, [&] {
        return std::string(op) + ": the system has no inputs; leave the input set empty";
    });
    CORA_CHECK(!U || U->center().shape() == std::vector<int64_t>({input_dim()}),
               Error::dimension_mismatch, [&] {
                   return std::string(op) + ": the input set must be one zonotope of the " +
                          std::to_string(input_dim()) + " inputs";
               });
}

DiscreteStep NonlinearSysDT::post(const Zonotope &R, const std::optional<Zonotope> &U,
                                  const alg::DiscreteLinearization &alg) const {
    // the checks, then the step of linReach
    detail::Entry entry;
    if (entry.check()) {
        validate_sets("post", R, U);
        alg.validate();
    }
    return priv_lin_reach(*this, R, U, alg);
}

ReachSetDT NonlinearSysDT::reach(const ReachProblem &problem,
                                 const alg::DiscreteLinearization &alg) const {
    detail::Entry entry;
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the initial set is required"); });
    const Zonotope &X0 = *problem.initial_set;
    if (entry.check()) {
        validate_sets("reach", X0, problem.input_set);
        alg.validate();
    }
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    // the times 0, dt, ..., as MATLAB's colon
    const int steps =
        static_cast<int>(std::floor(problem.t_final / time_step_ + numerics::kStepCount));
    const std::optional<Zonotope> U = priv_input_set(*this, X0.center(), problem.input_set);
    const auto zero = [&](int64_t d) {
        return Zonotope(X0.center().zeros_like({d}), X0.center().zeros_like({0, d}));
    };
    ReachSetDT R;
    R.time_step = time_step_;
    // the output set of a state set, or the states themselves
    const auto record = [&](const Zonotope &X, const Zonotope &error_x, bool first) {
        R.states.push_back(X);
        R.error_states.push_back(first ? zero(dim()) : error_x);
        DiscreteStep Y{X, zero(dim())};
        if (alg.compute_output_set) Y = output_set(X, U, alg);
        R.outputs.push_back(Y.set);
        R.error_outputs.push_back(Y.error);
    };
    record(X0, X0, true);
    Zonotope X = X0;
    for (int k = 0; k < steps; ++k) {
        const DiscreteStep next = priv_lin_reach(*this, X, U, alg);
        X = next.set;
        record(X, next.error, false);
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
