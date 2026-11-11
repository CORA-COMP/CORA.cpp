// linearize - the affine system of a step with y eliminated, as CORA's priv_linearize
//
// f and g are expanded at the point (x*, y*, u*): x* is the state half a step along the flow, y*
// the algebraic variables consistent with it (Newton from the guess), u* the center of the input
// set plus the constant input. Solving the linearized g for y puts the algebraic part into the
// state equation: x' = f0 + A_lin (x - x*) + B_lin (u - u*) with A_lin = A - C F^-1 D,
// B_lin = B - C F^-1 E and f0 = f - C F^-1 g [Althoff, Krogh, TAC 2014, Eqs. 13-15].
//
// Syntax:   DaeLinearized lin = sys.linearize(x0, y0, U, u_trans, time_step);
// Inputs:   x0, y0 - the center of the set and a guess of y there, U - the input set (optional),
//           u_trans - the constant input (m, 1), time_step - the step size
// Outputs:  lin - the expansion point, the blocks that the error needs, the affine system
// See also: consistent_initial_state, reach
//
// CORA: contDynamics/@nonlinDASys/private/priv_linearize.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

DaeLinearized NonlinDASys::linearize(const Tensor &x0, const Tensor &y0,
                                     const std::optional<Zonotope> &U, const Tensor &u_trans,
                                     double time_step) const {
    detail::Entry entry;
    CORA_CHECK(time_step > 0 && time_step < 1e300, Error::wrong_value, [&] {
        return "linearize: time_step must be positive and finite, got " + std::to_string(time_step);
    });
    CORA_CHECK(!U || m_ > 0, Error::not_supported,
               [] { return std::string("linearize: the system has no inputs; leave U empty"); });
    // the expansion point: the input's center, the state half a step on, y consistent with it
    const Tensor u_center = U ? U->center() : x0.zeros_like({m_, 1});
    const Tensor u_star = u_center + u_trans;
    const Tensor x_star = x0 + dynamics(x0, y0, u_star) * (0.5 * time_step);
    const Tensor y_star = newton_algebraic(x_star, y0, u_star, false);
    // the blocks of the Jacobian and the residuals there
    const Tensor f0_dyn = dynamics(x_star, y_star, u_star);
    const Tensor g0 = constraints(x_star, y_star, u_star);
    const DaeJacobians J = jacobian(x_star, y_star, u_star);
    // eliminating y: A_lin = A - C F^-1 D, B_lin = B - C F^-1 E, f0 = f - C F^-1 g
    const Tensor F_inv = J.F.pinv();
    const Tensor CF_inv = J.C.matmul(F_inv);
    const Tensor f0 = f0_dyn - CF_inv.matmul(g0);
    const Tensor A_lin = J.A - CF_inv.matmul(J.D);
    const Tensor B_lin = J.B - CF_inv.matmul(J.E);
    // the input of the affine system: B_lin (U - center), the constant part in u_const
    Zonotope BU(x0.zeros_like({n_, 1}), x0.zeros_like({n_, 0}));
    if (m_ > 0 && U) BU = (*U - u_center).linear_map(B_lin);
    return {x_star, y_star, u_star, g0, J.D, J.E, F_inv, CF_inv, LinearSys(A_lin), BU, f0};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
