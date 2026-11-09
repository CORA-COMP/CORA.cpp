// particular_solution_constant - what a constant input adds over one step
//
// For a constant u in U the set at the end of the step is A^-1 (e^{A dt} - I) U if A is
// invertible, else the series sum A^i dt^(i+1)/(i+1)! U of order taylor_terms plus its remainder.
// The curvature G U and the enclosure of the step conv(0, Ptp) + G U come with it. The origin
// stays the origin. Not a batch of systems: invertibility is decided for one matrix.
//
// Syntax:   ConstantSolution P = sys.particular_solution_constant(U, time_step, taylor_terms);
// Inputs:   U - input set in the state space (zonotope; a point for one input vector)
//           time_step - step size, positive
//           taylor_terms - order of the series, at least 1 (CORA's Inf is not offered)
// Outputs:  P - time_point (Ptp), input_curvature (C_input), time_interval (Pti)
// See also: particular_solution_time_varying, affine_solution, correction_matrix_input
//
// CORA: contDynamics/@linearSys/particularSolution_constant.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConstantSolution LinearSys::particular_solution_constant(const Zonotope &U, double time_step,
                                                         int taylor_terms) const {
    detail::Entry entry;
    if (entry.check()) validate_step("particular_solution_constant", U, time_step, taylor_terms);
    CORA_CHECK(A().ndim() == 2, Error::not_supported, [] {
        return std::string("particular_solution_constant: it supports a single system, not a "
                           "batch of matrices A");
    });
    TaylorLinSys taylor(A());
    BlockConstant P =
        priv_particular_constant(*this, taylor, priv_one_block(dim()), U, time_step, taylor_terms);
    return {P.time_point[0], P.input_curvature[0], P.time_interval[0]};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
