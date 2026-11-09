// particular_solution_time_varying - what a time-varying input adds over one step
//
// For u(t) in U the solution at the end of the step is the sum of A^i dt^(i+1)/(i+1)! U for
// i = 0..taylor_terms plus the remainder of the series; every term is kept as generators of
// its own, since u(t) takes a different value at each time. U may be off the origin: its center
// is part of the set.
//
// Syntax:   Zonotope PU = sys.particular_solution_time_varying(U, time_step, taylor_terms);
// Inputs:   U - input set in the state space (zonotope of the dimension of A)
//           time_step - step size, positive
//           taylor_terms - order of the series, at least 1 (CORA's Inf is not offered)
// Outputs:  PU - the origin if U is the origin
// See also: particular_solution_constant, one_step, expm_remainder
//
// CORA: contDynamics/@linearSys/particularSolution_timeVarying.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope LinearSys::particular_solution_time_varying(const Zonotope &U, double time_step,
                                                     int taylor_terms) const {
    detail::Entry entry;
    if (entry.check())
        validate_step("particular_solution_time_varying", U, time_step, taylor_terms);
    TaylorLinSys taylor(A());
    return priv_particular_time_varying(*this, taylor, priv_one_block(dim()), U, time_step,
                                        taylor_terms)[0];
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
