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

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The linear step
// ----------------------------------------------------------------------------------

/// CORALean's polyOneStepFloat at the closed-form linear step of the affine system at x*: both
/// sets of one step of x' = A x + u, u in V, from the polynomial zonotope R, shifted back by x*.
class PolyStepper {
  public:
    PolyStepper(const Tensor &A, const Tensor &xs, const PolyZonotope &R,
                const alg::Linearization &alg)
        : stepper_(A, alg.time_step, std::max(2, alg.taylor_terms)), xs_(xs),
          X_(priv_poly_plus(R, Zonotope(xs * -1.0, xs.zeros_like({xs.size(0), 0})))) {
        if (!stepper_.admissible())
            CORA_FAIL(Error::not_converged,
                      "cora: no admissible Taylor order for the linearized system (the time_step "
                      "may be too large)");
        // the parts that the input does not change: [e^{A dt}] X, [F] X and the hull of X and Htp
        // are formed with Pu, and Pu depends on the constant input, so they wait for the input
    }

    /// The sets over the step and at its end for the input set V.
    std::pair<PolyZonotope, PolyZonotope> operator()(const Zonotope &V) const {
        const Tensor u = V.center();
        const Zonotope Vc(u.zeros_like(), V.generators());
        const Zonotope point(u, u.zeros_like({u.size(0), 0}));
        const Zonotope PU = stepper_.input_solution(Vc);
        const Zonotope Pu = stepper_.input_solution(point);
        const Zonotope C_input = point.linear_map(stepper_.G());
        const PolyZonotope Htp = priv_poly_plus(priv_poly_interval_map(stepper_.e_adt(), X_), Pu);
        const PolyZonotope C_state = priv_poly_interval_map(stepper_.F(), X_);
        const PolyZonotope Hti =
            priv_poly_plus(priv_poly_construct(priv_poly_enclose_shared(X_, Htp)), C_state);
        const PolyZonotope Rtp = priv_poly_plus(Htp, PU);
        const PolyZonotope Rti = priv_poly_plus(priv_poly_plus(Hti, PU), C_input);
        return {back(Rti), back(Rtp)};
    }

  private:
    PolyZonotope back(const PolyZonotope &Z) const {
        return priv_poly_plus(Z, Zonotope(xs_, xs_.zeros_like({xs_.size(0), 0})));
    }
    ClosedFormStepper stepper_;
    Tensor xs_;
    PolyZonotope X_;
};

/// The step of CORALean's linReachPolyLoopFloat: CORA's loop of linReach.m on a polynomial
/// zonotope, accepted when the error is at most the applied one entry by entry.
PolyNonlinearStep aux_lean_step(const NonlinearSys &sys, const PolyZonotope &R, const Tensor &error,
                                const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    constexpr double enlarge = 1.1; // 2476979795053773 * 2^-51
    constexpr int fuel = 10000;
    const Tensor &like = R.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    const std::optional<Zonotope> Uz =
        m > 0 ? std::optional<Zonotope>(
                    U ? *U : Zonotope(like.zeros_like({m, 1}), like.zeros_like({m, 0})))
              : std::nullopt;
    // x*, A, B and f(x*, u*) + B (U - u*)
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, Uz, like, alg.time_step);
    const Vec us = Uz ? Uz->center().data() : Vec{};
    const std::optional<Box> Ub = Uz ? std::optional<Box>(priv_zonotope_box(*Uz)) : std::nullopt;
    const std::optional<Box> Uh = Ub ? std::optional<Box>(priv_hull(*Ub, us)) : std::nullopt;
    const PolyStepper step(L.J.A, L.xs_t, R, alg);

    Vec abstrerr = error.data();
    for (int pass = 0; pass < fuel; ++pass) {
        // the applied error, and the time-interval set of the step with it in the input
        Box applied{Vec(abstrerr.size()), Vec(abstrerr.size())};
        for (std::size_t i = 0; i < abstrerr.size(); ++i) {
            applied.hi[i] = enlarge * abstrerr[i];
            applied.lo[i] = -applied.hi[i];
        }
        const Zonotope Rmax =
            priv_poly_zonotope_fold(step(L.Vbase + priv_box_zonotope(like, applied)).first);
        // the Hessian remainder over Rmax joined with x*, its magnitude
        const Box Xh = priv_hull(priv_zonotope_box(Rmax), L.xs);
        const Zonotope Verror = priv_box_zonotope(
            like, priv_hessian_remainder(priv_hessian(sys, like, Xh, Uh), {}, Xh, Uh, L.xs, us, n));
        const Vec true_error = priv_norm_mag(priv_zonotope_box(Verror));
        priv_check_exploded(priv_column(like, true_error));
        bool holds = true;
        for (std::size_t i = 0; i < true_error.size(); ++i)
            holds = holds && true_error[i] <= applied.hi[i];
        if (holds) {
            // the sets with the accepted remainder in the input
            auto [Rti, Rtp] = step(L.Vbase + Verror);
            return {std::move(Rti), std::move(Rtp), priv_column(like, true_error)};
        }
        abstrerr = true_error;
    }
    CORA_FAIL(Error::not_converged, "cora: the error loop of the step did not converge");
}

} // namespace


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
    // CORALean's step at tensor order 2 with a linear step of CORALean's, bit for bit at nearest
    if (alg.linear_step != alg::LinearStep::cora && alg.tensor_order == 2)
        return aux_lean_step(sys, R, error, U, alg);
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
