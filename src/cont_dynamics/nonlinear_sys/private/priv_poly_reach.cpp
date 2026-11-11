// priv_poly_reach - one step of a nonlinear system by polynomialization, as CORA's linReach "poly"
//
// As priv_lin_reach, but the set is a polynomial zonotope. The step linearizes at the center
// (x*, u*), solves the affine system from R - x* in a polynomial zonotope (the exponential e^{A dt}
// maps it exactly, the curvature adds F R as an enclosure), and adds the solution of the error.
// The error has a static part, the Hessian at the expansion point mapped over R - x* as a
// polynomial zonotope, and a dynamic part, a zonotope over the set of the differences Rdiff of the
// states over the step to its start (priv_aux_delta_reach). The dynamic part is guessed as 1.1
// times the error of the step before and enlarged until it holds, as in lin. Both solutions are
// added with exact_plus, so the dependency of the static one on the initial set stays.
//
// Syntax:   PolyNonlinearStep s = priv_poly_reach(sys, R, error, U, alg);
// Inputs:   R - the set at the start
//           error - the error bound of the step before
//           U - the input set (uncentered, optional)
//           alg - the options of "poly"
// Outputs:  s - the sets over the step and at its end, and the error bound of this step
// See also: priv_lin_reach, priv_abstrerr_poly, priv_precomp_stat_error
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The solution at the end of the step of a constant polynomial zonotope input U (CORA's
/// particularSolution_constant): A^-1 (e^{A dt} - I) U if A is invertible, else the series of A
/// and the remainder E of the exponential, which add on factors of their own.
PolyZonotope aux_constant_input(const LinearSys &sys, TaylorLinSys &taylor, const PolyZonotope &U,
                                double dt, int eta) {
    if (const std::optional<Tensor> Ainv = taylor.Ainv())
        return U.linear_map(Ainv->matmul(taylor.e_adt(dt) - sys.A().eye_like()));
    const Tensor Asum = priv_series(sys, taylor, dt, eta);
    return U.linear_map(Asum).plus(U.linear_map(priv_remainder(sys, dt, eta)));
}

/// The set of the differences of the states over the step to its start (CORA's aux_deltaReach):
/// the hull of the origin and (e^{A dt} - I) R + Pu, the curvature F R and the input
/// curvature, reduced, plus the reduced input solution PU. The origin keeps the factors of R.
PolyZonotope aux_delta_reach(const LinearSys &sys, TaylorLinSys &taylor, const PolyZonotope &R,
                             const Zonotope &PU, const Zonotope &Pu, const Zonotope &C_input,
                             double dt, int eta, int order) {
    const Tensor e_at = taylor.e_adt(dt);
    const PolyZonotope end = R.linear_map(e_at - e_at.eye_like()).plus(Pu);
    // 0 * end: the same factors, no generators
    const Tensor zero = R.center().zeros_like();
    const PolyZonotope origin(zero, zero.zeros_like({R.dim(), 0}), zero.zeros_like({R.dim(), 0}),
                              {}, end.ids());
    const Interval F = sys.correction_matrix_state(dt, eta);
    const PolyZonotope hom = origin.enclose(end).plus(R.zonotope().linear_map(F)).plus(C_input);
    return hom.reduce(order).plus(PU.reduce(order));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The sets of the difference and the error --------------------------------------------------------

// The step ----------------------------------------------------------------------------------------

PolyNonlinearStep priv_poly_reach(const NonlinearSys &sys, const PolyZonotope &R,
                                  const Tensor &error, const std::optional<Zonotope> &U,
                                  const alg::Polynomialization &alg) {
    const double dt = alg.time_step;
    const int eta = alg.taylor_terms;
    // the affine system and the pieces of its step; the powers of A are shared
    PolyStepPieces s(sys, R, U, dt, eta);
    // the differences over the step, and the static error, which the loop does not change
    const PolyZonotope Rdiff =
        aux_delta_reach(s.lin.sys, s.taylor, s.R_delta, s.PU, s.Pu.time_point[0],
                        s.Pu.input_curvature[0], dt, eta, alg.intermediate_order);
    const PolyStaticError stat =
        priv_precomp_stat_error(sys, s.R_delta, s.lin, sys.input_dim() > 0, alg);
    // the error: guessed, checked over the set it allows, and enlarged until it holds
    const AbstractionError err =
        priv_error_loop(s.lin, s.taylor, error, dt, eta, [&](const Zonotope &R_guess) {
            const PolyZonotope R_max = s.R_delta.plus(Rdiff.zonotope()).plus(R_guess);
            const PolyAbstractionError e =
                priv_abstrerr_poly(sys, R_max, Rdiff.plus(R_guess), s.lin, s.U0, stat, alg);
            return AbstractionError{e.bound, e.set};
        });
    // the solutions of the dynamic and the static error; exact_plus keeps the dependency
    const Zonotope R_dyn =
        priv_particular_time_varying(s.lin.sys, s.taylor, s.blocks, err.set, dt, eta)[0];
    const PolyZonotope R_error =
        aux_constant_input(s.lin.sys, s.taylor, stat.error, dt, eta).plus(R_dyn);
    return {s.R_ti.plus(s.lin.x_star).exact_plus(R_error),
            s.R_tp.plus(s.lin.x_star).exact_plus(R_error), err.bound};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
