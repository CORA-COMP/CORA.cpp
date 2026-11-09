// one_step - one step of reach from a set, as CORA's oneStep
//
// The reachable sets at the end of the step and over it, from X with the time-varying input
// set U and the constant input u: the homogeneous solution shifted by the constant input, plus
// the time-varying input solution and the two curvatures. A building block of every reach
// algorithm.
//
// Syntax:   OneStep s = sys.one_step(X, U, u, time_step, taylor_terms);
// Inputs:   X - initial set of the step (zonotope of the dimension of A)
//           U - input set in the state space, centered (zonotope; the origin for none)
//           u - constant input in the state space (n, 1)
//           time_step - step size, positive
//           taylor_terms - order of the series, at least 1
// Outputs:  s - time_point (Rtp), time_interval (Rti) and the parts they are made of
// See also: homogeneous_solution, particular_solution_time_varying, particular_solution_constant
//
// CORA: contDynamics/@linearSys/oneStep.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

OneStep LinearSys::one_step(const Zonotope &X, const Zonotope &U, const Tensor &u, double time_step,
                            int taylor_terms) const {
    detail::Entry entry;
    if (entry.check()) validate_step("one_step", X, time_step, taylor_terms);
    // the input set and the constant input live in the state space
    CORA_CHECK(U.dim() == dim() && u.shape() == std::vector<int64_t>({dim(), 1}),
               Error::dimension_mismatch, [&] {
                   return "one_step: U must have the dimension " + std::to_string(dim()) +
                          " and u the shape (" + std::to_string(dim()) + ", 1), got " +
                          std::to_string(U.dim()) + " and " + shape_text(u.shape());
               });
    CORA_CHECK(A().ndim() == 2, Error::not_supported, [] {
        return std::string("one_step: it supports a single system, not a batch of matrices A");
    });
    // the powers of A are shared by the solutions; the constant input is a point set
    TaylorLinSys taylor(A());
    const Zonotope point(u, u.zeros_like({dim(), 0}));
    BlockStep s =
        priv_one_step(*this, taylor, priv_one_block(dim()), X, U, point, time_step, taylor_terms);
    return {s.time_point[0],
            s.time_interval[0],
            s.homogeneous_time_point[0],
            s.homogeneous_time_interval[0],
            s.time_varying[0],
            s.constant[0],
            s.state_curvature[0],
            s.input_curvature[0]};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
