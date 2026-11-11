// linearize - the affine system of one step of a nonlinear system, as CORA's linearize
//
// The expansion point is the center c of R moved by half a step along the flow,
//     x* = c + f(c, u*) dt/2,   u* = the center of U,
// and the step solves x' = A (x - x*) + f(x*, u*) + B (u - u*) with A, B the Jacobians at
// (x*, u*). As in CORA the affine system is the LinearSys x' = A x + u whose input u lies in the
// set B (U - u*) (centered) plus the constant f(x*, u*) and that center.
//
// Syntax:   Linearized lin = sys.linearize(R, U, time_step);
// Inputs:   R - the set of the step's start, U - the input set (none: the input 0), time_step
// Outputs:  lin - x_star, u_star, f0, the linear system and its input
// See also: reach, init_reach
//
// CORA: contDynamics/@nonlinearSys/linearize.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Linearized NonlinearSys::linearize(const Zonotope &R, const std::optional<Zonotope> &U,
                                   double time_step) const {
    detail::Entry entry;
    if (entry.check()) validate_sets("linearize", R, U);
    CORA_CHECK(time_step > 0 && time_step < 1e300, Error::wrong_value, [&] {
        return "linearize: time_step must be positive and finite, got " + std::to_string(time_step);
    });
    return linearize_impl(R.center(), U, time_step);
}

Linearized NonlinearSys::linearize_impl(const Tensor &center, const std::optional<Zonotope> &U,
                                        double time_step) const {
    // the expansion point: the input set's center, the set's center half a step on
    const Tensor &c = center;
    const std::optional<Tensor> u_star =
        m_ == 0 ? std::nullopt : std::optional<Tensor>(U ? U->center() : c.zeros_like({m_, 1}));
    const Tensor x_star = c + dynamics(c, u_star) * (0.5 * time_step);
    const Tensor f0 = dynamics(x_star, u_star);
    const Jacobians J = jacobian(x_star, u_star);
    // the input of the affine system: B (U - u*) around its center, and the constant part
    Zonotope BU(c.zeros_like({n_, 1}), c.zeros_like({n_, 0}));
    if (m_ > 0 && U) BU = (*U - *u_star).linear_map(J.B);
    const Tensor u_center = BU.center();
    return {
        x_star,       u_star ? *u_star : c.zeros_like({0, 1}), f0, LinearSys(J.A), BU - u_center,
        f0 + u_center};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
