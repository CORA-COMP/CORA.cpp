// priv_affine_step - one step of the affine system of a linearization, as linReach's oneStep
//
// The affine system x' = A x + u is solved from R - x* over one step by LinearSys's block step. The
// powers of A and the blocks it computed are returned with the sets, because the solutions of the
// linearization error (particularSolution_timeVarying) of the same step reuse them. The centered
// input set and the check of a linearization error that exploded are the other two parts that the
// steps of nonlinear and of algebraic systems have in common.
//
// Syntax:   AffineStep s = priv_affine_step(sys, R_delta, U, u_const, time_step, taylor_terms);
// Inputs:   sys - the affine system, R_delta - the set relative to x*, U - the input set,
//           u_const - the constant input, time_step, taylor_terms - the step
// Outputs:  s - the sets over the step and at its end, the Taylor data of A and the blocks
// See also: priv_lin_reach, priv_dae_lin_reach, LinearSys::one_step
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

AffineStep priv_affine_step(const LinearSys &sys, const Zonotope &R_delta, const Zonotope &U,
                            const Tensor &u_const, double time_step, int taylor_terms) {
    // the powers of A are shared by every solution of the step
    const int64_t n = R_delta.center().shape()[0];
    TaylorLinSys taylor(sys.A());
    Blocks blocks = priv_one_block(n);
    const Zonotope u(u_const, u_const.zeros_like({n, 0}));
    BlockStep step = priv_one_step(sys, taylor, blocks, R_delta, U, u, time_step, taylor_terms);
    return {std::move(taylor), std::move(blocks), std::move(step.time_interval[0]),
            std::move(step.time_point[0])};
}

std::optional<Zonotope> priv_centered_inputs(const Tensor &like, int64_t m,
                                             const std::optional<Zonotope> &U) {
    if (m == 0) return std::nullopt;
    return U ? *U - U->center() : Zonotope(like.zeros_like({m, 1}), like.zeros_like({m, 0}));
}

void priv_check_exploded(const Tensor &error) {
    if (priv_exploded(error))
        CORA_FAIL(Error::reach_set_explosion,
                  "cora: the linearization error exploded; the reachable set grows without "
                  "bound (the time_step may be too large)");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
