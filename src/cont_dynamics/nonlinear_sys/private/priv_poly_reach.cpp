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
    const int64_t n = sys.dim();
    // the affine system and the pieces of its step; the powers of A are shared
    const Linearized lin = sys.linearize_impl(R.center(), U, dt);
    const PolyZonotope R_delta = R.plus(lin.x_star * -1.0);
    TaylorLinSys taylor(lin.sys.A());
    const Blocks blocks = priv_one_block(n);
    const Zonotope u_const(lin.u_const, lin.u_const.zeros_like({n, 0}));
    const Zonotope PU = priv_particular_time_varying(lin.sys, taylor, blocks, lin.U, dt, eta)[0];
    const BlockConstant Pu = priv_particular_constant(lin.sys, taylor, blocks, u_const, dt, eta);
    // the homogeneous solution at the end, the hull of the start and it, and the state curvature
    const PolyZonotope H_tp = R_delta.linear_map(taylor.e_adt(dt)).plus(Pu.time_point[0]);
    const PolyZonotope C_state = R_delta.linear_map(lin.sys.correction_matrix_state(dt, eta));
    const PolyZonotope R_tp = H_tp.plus(PU);
    const PolyZonotope R_ti = R_delta.enclose(H_tp).plus(C_state).plus(PU + Pu.input_curvature[0]);
    // the differences over the step, and the static error, which the loop does not change
    const PolyZonotope Rdiff =
        aux_delta_reach(lin.sys, taylor, R_delta, PU, Pu.time_point[0], Pu.input_curvature[0], dt,
                        eta, alg.intermediate_order);
    const bool has_inputs = sys.input_dim() > 0;
    const PolyStaticError stat = priv_precomp_stat_error(sys, R_delta, lin, has_inputs, alg);
    // the centered input set (CORA's params.U); none for a system without inputs
    std::optional<Zonotope> U0;
    if (const int64_t m = sys.input_dim(); m > 0)
        U0 = U ? *U - U->center()
               : Zonotope(R.center().zeros_like({m, 1}), R.center().zeros_like({m, 0}));

    // the error: guessed, checked over the set it allows, and enlarged until it holds
    Tensor bound = error;
    std::optional<PolyAbstractionError> err;
    double perf = std::numeric_limits<double>::infinity();
    while (perf > 1) {
        const Tensor applied = bound * 1.1;
        const Zonotope V_guess(applied.zeros_like(), applied.diag());
        const Zonotope R_guess =
            priv_particular_time_varying(lin.sys, taylor, blocks, V_guess, dt, eta)[0];
        const PolyZonotope R_max = R_delta.plus(Rdiff.zonotope()).plus(R_guess);
        err = priv_abstrerr_poly(sys, R_max, Rdiff.plus(R_guess), lin, U0, stat, alg);
        perf = priv_ratio_max(err->bound, applied);
        bound = err->bound;
        if (priv_exploded(bound))
            CORA_FAIL(Error::reach_set_explosion,
                      "cora: the linearization error exploded; the reachable set grows without "
                      "bound (the time_step may be too large)");
    }
    // the solutions of the dynamic and the static error; exact_plus keeps the dependency
    const Zonotope R_dyn =
        priv_particular_time_varying(lin.sys, taylor, blocks, err->set, dt, eta)[0];
    const PolyZonotope R_error =
        aux_constant_input(lin.sys, taylor, stat.error, dt, eta).plus(R_dyn);
    return {R_ti.plus(lin.x_star).exact_plus(R_error), R_tp.plus(lin.x_star).exact_plus(R_error),
            bound};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
