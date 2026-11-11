// priv_lin_reach_poly - one step of "lin" from a polynomial zonotope, as CORA's linReach
//
// As priv_lin_reach, but the set is a polynomial zonotope, so the dependency on the initial set
// stays through the linear part: e^{A dt} maps R - x* exactly, the hull with the start set and
// the curvature F R are enclosures that keep the factors. The linearization error is the one of
// "lin" (priv_abstrerr_lin, over the zonotope that encloses the step), a zonotope, and it joins
// both sets as independent generators (CORA's Rti + Rerror). The linearization point is the
// center c of the representation, which is a point of the set (all factors zero); the region of
// the Hessian still joins the box with it (priv_abstrerr_lin).
//
// Syntax:   PolyNonlinearStep s = priv_lin_reach_poly(sys, R, error, U, alg);
// Inputs:   R - the polynomial zonotope at the start of the step, error - the bound of the step
//           before, U - the input set (uncentered, optional), alg - the options of "lin"
// Outputs:  s - the sets over the step and at its end, and the error bound of this step
// See also: priv_lin_reach, priv_poly_reach
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


// ===========================================  MAIN  =========================================== //

PolyStepPieces::PolyStepPieces(const NonlinearSys &sys, const PolyZonotope &R,
                               const std::optional<Zonotope> &U, double dt, int eta)
    : lin(sys.linearize_impl(R.center(), U, dt)), R_delta(R.plus(lin.x_star * -1.0)),
      taylor(lin.sys.A()), blocks(priv_one_block(sys.dim())),
      PU(priv_particular_time_varying(lin.sys, taylor, blocks, lin.U, dt, eta)[0]),
      Pu(priv_particular_constant(lin.sys, taylor, blocks,
                                  Zonotope(lin.u_const, lin.u_const.zeros_like({sys.dim(), 0})), dt,
                                  eta)),
      R_tp(R_delta.linear_map(taylor.e_adt(dt)).plus(Pu.time_point[0]).plus(PU)),
      R_ti(R_delta.enclose(R_delta.linear_map(taylor.e_adt(dt)).plus(Pu.time_point[0]))
               .plus(R_delta.linear_map(lin.sys.correction_matrix_state(dt, eta)))
               .plus(PU + Pu.input_curvature[0])) {
    // the centered input set (CORA's params.U); none for a system without inputs
    if (const int64_t m = sys.input_dim(); m > 0)
        U0 = U ? *U - U->center()
               : Zonotope(R.center().zeros_like({m, 1}), R.center().zeros_like({m, 0}));
}

PolyNonlinearStep priv_lin_reach_poly(const NonlinearSys &sys, const PolyZonotope &R,
                                      const Tensor &error, const std::optional<Zonotope> &U,
                                      const alg::Linearization &alg) {
    const double dt = alg.time_step;
    const int eta = alg.taylor_terms;
    PolyStepPieces s(sys, R, U, dt, eta);
    // the error: guessed, checked over the set it allows, and enlarged until it holds
    const AbstractionError err =
        priv_error_loop(s.lin, s.taylor, error, dt, eta, [&](const Zonotope &R_guess) {
            return priv_abstrerr_lin(sys, s.R_ti.plus(R_guess).zonotope(), s.lin, s.U0, alg);
        });
    // the solution of the error joins both sets as independent generators
    const Zonotope R_error =
        priv_particular_time_varying(s.lin.sys, s.taylor, s.blocks, err.set, dt, eta)[0];
    return {s.R_ti.plus(s.lin.x_star).plus(R_error), s.R_tp.plus(s.lin.x_star).plus(R_error),
            err.bound};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
