// consistent_initial_state - the y with g(x0, y, u0) = 0, as CORA's consistentInitialState
//
// Newton's method on g in y from the guess y0: delta_y solves F delta_y = -g with F = dg/dy at the
// current y, and delta_y is added also when it is the last one (its 2-norm below 1e-10), as CORA's
// loop does. The helper of CORA's priv_linearize is the same loop that stops before adding that
// step (newton_algebraic with add_last false). CORA's loops have no bound; these stop after 100
// steps with not_converged.
//
// Syntax:   Tensor y = sys.consistent_initial_state(x0, y0, u0);
// Inputs:   x0 - the state (n, 1), y0 - the guess of y (p, 1), u0 - the input (m, 1)
// Outputs:  y - the algebraic variables after the last Newton step (p, 1)
// See also: steady_state, linearize
//
// CORA: contDynamics/@nonlinDASys/consistentInitialState.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor NonlinDASys::consistent_initial_state(const Tensor &x0, const Tensor &y0,
                                             const Tensor &u0) const {
    detail::Entry entry;
    if (entry.check()) validate_state("consistent_initial_state", x0, y0);
    return newton_algebraic(x0, y0, u0, true);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
