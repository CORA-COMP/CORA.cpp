// steady_state - the state and algebraic variables with f = 0 and g = 0, as CORA's steadyState
//
// Newton's method on the stacked system (f, g) in (x, y) at the input u0 from (x0, y0): the step
// solves [A C; D F] delta_z = -(f, g) at the current point, and it is added also when it is the
// last one (its 2-norm below 1e-10). CORA's loop has no bound; this one stops after 100 steps.
//
// Syntax:   auto [x, y] = sys.steady_state(x0, y0, u0);
// Inputs:   x0 - the state (n, 1), y0 - the guess of y (p, 1), u0 - the input (m, 1)
// Outputs:  x, y - the point after the last Newton step
// See also: consistent_initial_state
//
// CORA: contDynamics/@nonlinDASys/steadyState.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> NonlinDASys::steady_state(const Tensor &x0, const Tensor &y0,
                                                    const Tensor &u0) const {
    detail::Entry entry;
    if (entry.check()) validate_state("steady_state", x0, y0);
    Tensor x = x0, y = y0;
    for (int i = 0; i < 100; ++i) {
        // the residual (f, g) and the matrix [A C; D F] of the stacked system
        const Tensor k = dynamics(x, y, u0), l = constraints(x, y, u0);
        const DaeJacobians J = jacobian(x, y, u0);
        const Tensor M =
            Tensor::concat({Tensor::concat({J.A, J.C}, -1), Tensor::concat({J.D, J.F}, -1)}, -2);
        const Tensor delta_z = M.solve(-Tensor::concat({k, l}, -2));
        x = x + delta_z.slice(-2, 0, n_);
        y = y + delta_z.slice(-2, n_, n_ + p_);
        if (delta_z.vector_norm(Tensor::Norm::two, -2).item() < numerics::kNewtonStep)
            return {x, y};
    }
    CORA_FAIL(Error::not_converged, "steady_state: Newton's method did not converge in 100 steps");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
