// priv_lin_reach - one step of a nonlinear system by conservative linearization, as CORA's linReach
//
// The step linearizes at (x*, u*) (linearize), solves the affine system from R - x* by one step
// of LinearSys (oneStep), and adds the solution of the linearization error L. L is guessed as 1.1
// times the error of the step before (0 at the first), the set of the step is enlarged by the
// solution of the guess, and the error over that set is computed (priv_abstrerr_lin). While it
// exceeds the guess somewhere, the guess becomes the error; then its solution is added to both
// sets. MATLAB's max skips NaN: an error of 0 over a guess of 0 does not count, and all-NaN ends
// the loop.
//
// With a linear step of CORALean's at tensor order 2 the step is CORALean's, bit for bit at
// nearest. error_bound = loop is linReachLoopFloat: 1.1 times the last error is the input [-g, g]
// of one linear step, the Hessian over its set (with the expansion point) is the error, accepted
// when it is at most g entry by entry, else the new guess; the sets are one step with it as input.
// error_bound = region is linReachFloat: a region checked once replaces the guess and the loop.
//
// Syntax:   NonlinearStep s = priv_lin_reach(sys, R, error, U, alg);
// Inputs:   R - the set at the start of the step, error - the error bound of the step before
//           U - the input set (uncentered), alg - the options of "lin"
// Outputs:  s - the sets over the step and at its end, and the error bound of this step
// See also: priv_abstrerr_lin, NonlinearSys::linearize, LinearSys::one_step
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_closed_form_step.h"
#include "cont_dynamics/linear_sys/private/priv_float_step.h"
#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The linear step of the input set V from R (the sets of the window and of the end of the step),
/// CORALean's: reachFloatInput (coralean) or CORA's oneStep in closed form (closed_form).
using LinearStepFn = std::function<FloatStep(const Zonotope &V)>;

// The linear step ---------------------------------------------------------------------------------

/// The linear step of the affine system at x* from R under the input set V (oneStepFloat):
/// CORALean's reachFloatInput or CORA's oneStep in closed form; the matrices once.
class LeanStepper {
  public:
    LeanStepper(const Tensor &A, const Tensor &xs, const Zonotope &R, const alg::Linearization &alg)
        : xs_(xs), R_(R), order_(alg.zonotope_order),
          closed_form_(alg.linear_step == alg::LinearStep::closed_form) {
        if (closed_form_)
            closed_.emplace(A, alg.time_step, std::max(2, alg.taylor_terms)); // CORALean: >= 2
        else
            lean_.emplace(A, alg.time_step, alg.taylor_terms);
        if (!(closed_form_ ? closed_->admissible() : lean_->admissible()))
            CORA_FAIL(Error::not_converged,
                      "cora: no admissible Taylor order for the linearized system (the time_step "
                      "may be too large)");
    }

    /// The sets over the step and at its end, for the input set V.
    FloatStep operator()(const Zonotope &V) const {
        if (!closed_form_) return lean_->step(xs_, V, R_, order_);
        // the constant input is the center of V, the time-varying one V without it
        const ClosedFormStep S = closed_->step(R_ - xs_, V - V.center(), V.center());
        return FloatStep{S.time_interval + xs_, S.time_point + xs_};
    }

  private:
    Tensor xs_;
    Zonotope R_;
    int order_;
    bool closed_form_;
    std::optional<FloatStepper> lean_;
    std::optional<ClosedFormStepper> closed_;
};

// The region step ---------------------------------------------------------------------------------

/// max over the coordinates of max(|lo|, |hi|), the bound of CORALean's regionFloat; NaN if one is.
double aux_bound(const Box &b) {
    double out = 0;
    for (std::size_t i = 0; i < b.lo.size(); ++i) {
        const double v = std::max(std::abs(b.lo[i]), std::abs(b.hi[i]));
        if (std::isnan(v)) return v;
        out = std::max(out, v);
    }
    return out;
}

/// The region the step stays in (regionFloat): X + [-delta, delta] at delta = (2 M0 + 1) dt, if
/// the bound M of f over it and U keeps M dt < delta, then two Picard iterates; none otherwise.
std::optional<Box> aux_region(const NonlinearSys &sys, const Tensor &like, const Box &X,
                              const std::optional<Box> &Ub, double dt) {
    const double M0 = aux_bound(priv_enclosure(sys, like, X, Ub));
    const double delta = ((M0 + M0) + 1.0) * dt;
    Box Omega = X;
    for (std::size_t i = 0; i < X.lo.size(); ++i) {
        Omega.lo[i] = X.lo[i] - delta;
        Omega.hi[i] = X.hi[i] + delta;
    }
    const double M = aux_bound(priv_enclosure(sys, like, Omega, Ub));
    if (!(M * dt < delta)) return std::nullopt;
    // X + [0, dt] f(Omega, U), twice
    const auto picard = [&](const Box &O) {
        const Box F = priv_enclosure(sys, like, O, Ub);
        Box out = X;
        for (std::size_t i = 0; i < X.lo.size(); ++i) {
            out.lo[i] = X.lo[i] + std::min(0.0, dt * F.lo[i]);
            out.hi[i] = X.hi[i] + std::max(0.0, dt * F.hi[i]);
        }
        return out;
    };
    return picard(picard(Omega));
}

/// The step as CORALean's linReachFloat at 'lin': the region, the Hessian remainder over it, and
/// one linear step with f(x*, u*) + B (U - u*) + [-l, l] as input.
NonlinearStep aux_region_step(const NonlinearSys &sys, const Zonotope &R,
                              const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    const Tensor &like = R.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    // without an input set the input is the origin
    const std::optional<Zonotope> Uz =
        m > 0 ? std::optional<Zonotope>(
                    U ? *U : Zonotope(like.zeros_like({m, 1}), like.zeros_like({m, 0})))
              : std::nullopt;
    const Box X = priv_zonotope_box(R);
    const std::optional<Box> Ub = Uz ? std::optional<Box>(priv_zonotope_box(*Uz)) : std::nullopt;
    const std::optional<Box> Omega = aux_region(sys, like, X, Ub, alg.time_step);
    if (!Omega)
        CORA_FAIL(Error::not_converged,
                  "cora: the region check of the step failed (the time_step may be too large)");
    // x*, A, B and f(x*, u*) + B (U - u*)
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, Uz, like, alg.time_step);
    const Vec us = Uz ? Uz->center().data() : Vec{};
    // the Hessian over the region and U, both holding the expansion point
    const Box Xh = priv_hull(*Omega, L.xs);
    const std::optional<Box> Uh = Uz ? std::optional<Box>(priv_hull(*Ub, us)) : std::nullopt;
    const Box err =
        priv_hessian_remainder(priv_hessian(sys, like, Xh, Uh), {}, Xh, Uh, L.xs, us, n);
    const LeanStepper step(L.J.A, L.xs_t, R, alg);
    const FloatStep S = step(L.Vbase + priv_box_zonotope(like, err));
    return {S.first, S.second, priv_column(like, err.hi)};
}

/// The step as CORALean's linReachLoopFloat: CORA's loop of linReach.m, accepted when the error is
/// at most the applied one entry by entry (floats, not their quotient); NaN repeats, fuel fails.
NonlinearStep aux_loop_step(const NonlinearSys &sys, const Zonotope &R, const Tensor &error,
                            const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    constexpr double enlarge = 1.1; // 2476979795053773 * 2^-51
    constexpr int fuel = 10000;
    const Tensor &like = R.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    const std::optional<Zonotope> Uz =
        m > 0 ? std::optional<Zonotope>(
                    U ? *U : Zonotope(like.zeros_like({m, 1}), like.zeros_like({m, 0})))
              : std::nullopt;
    // x*, A, B and f(x*, u*) + B (U - u*), as the region step
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, Uz, like, alg.time_step);
    const Vec us = Uz ? Uz->center().data() : Vec{};
    const std::optional<Box> Ub = Uz ? std::optional<Box>(priv_zonotope_box(*Uz)) : std::nullopt;
    // U joined with the expansion point
    const std::optional<Box> Uh = Ub ? std::optional<Box>(priv_hull(*Ub, us)) : std::nullopt;
    const LeanStepper step(L.J.A, L.xs_t, R, alg);

    Vec abstrerr = error.data();
    for (int pass = 0; pass < fuel; ++pass) {
        // the applied error, and the time-interval set of the step with it in the input
        Box applied{Vec(abstrerr.size()), Vec(abstrerr.size())};
        for (std::size_t i = 0; i < abstrerr.size(); ++i) {
            applied.hi[i] = enlarge * abstrerr[i];
            applied.lo[i] = -applied.hi[i];
        }
        const Zonotope Rmax = step(L.Vbase + priv_box_zonotope(like, applied)).first;
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
            const FloatStep S = step(L.Vbase + Verror);
            return {S.first, S.second, priv_column(like, true_error)};
        }
        abstrerr = true_error;
    }
    CORA_FAIL(Error::not_converged, "cora: the error loop of the step did not converge");
}

/// The step with a linear step of CORALean's (alg.linear_step): the guess of the error enters the
/// input of the step, and the accepted error is part of the input of the step that is returned.
NonlinearStep aux_lean_step(const NonlinearSys &sys, const Zonotope &R, const Tensor &error,
                            const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    const double dt = alg.time_step;
    // the affine system; its input without the error is f(x*, u*) + B (U - u*), centered at f
    const Linearized lin = sys.linearize(R, U, dt);
    const Zonotope V_base(lin.u_const, lin.U.generators());
    const LeanStepper step(lin.sys.A(), lin.x_star, R, alg);
    // the centered input set (CORA's params.U); without one, the input 0
    const std::optional<Zonotope> U0 = priv_centered_inputs(R.center(), sys.input_dim(), U);

    // the error: guessed, checked over the window it allows, and enlarged until it holds
    Tensor bound = error;
    std::optional<AbstractionError> err;
    double perf = std::numeric_limits<double>::infinity();
    while (perf > 1) {
        const Tensor applied = bound * 1.1;
        const Zonotope V_guess(applied.zeros_like(), applied.diag());
        const Zonotope window = step(V_base + V_guess).first;
        // the error is computed relative to the point of linearization
        const Zonotope relative(window.center() - lin.x_star, window.generators());
        err = priv_abstrerr_lin(sys, relative, lin, U0, alg);
        perf = priv_ratio_max(err->bound, applied);
        bound = err->bound;
        priv_check_exploded(bound);
    }
    // the error is part of the input of the step that returns the sets
    const FloatStep S = step(V_base + err->set);
    return {S.first, S.second, bound};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The ratios of the errors ------------------------------------------------------------------------

double priv_ratio_max(const Tensor &true_error, const Tensor &applied) {
    const std::vector<double> e = true_error.data(), a = applied.data();
    double out = std::numeric_limits<double>::quiet_NaN();
    // a 0/0 entry is skipped, as MATLAB's max skips NaN
    for (std::size_t i = 0; i < e.size(); ++i) {
        const double r = e[i] / a[i];
        if (!std::isnan(r) && (std::isnan(out) || r > out)) out = r;
    }
    return out;
}

bool priv_exploded(const Tensor &error) {
    for (const double v : error.data())
        if (v > 1e100) return true;
    return false;
}

// The loop and the step ---------------------------------------------------------------------------

AbstractionError
priv_error_loop(const Linearized &lin, TaylorLinSys &taylor, const Tensor &error, double dt,
                int eta,
                const std::function<AbstractionError(const Zonotope &R_guess)> &error_over) {
    // the sets of the guess live on one block of the state
    const Blocks blocks = priv_one_block(error.shape()[0]);
    Tensor bound = error;
    std::optional<AbstractionError> err;
    double perf = std::numeric_limits<double>::infinity();
    // guess 1.1 times the last error, solve it over the step, check the error over that set
    while (perf > 1) {
        const Tensor applied = bound * 1.1;
        const Zonotope V_guess(applied.zeros_like(), applied.diag());
        const Zonotope R_guess =
            priv_particular_time_varying(lin.sys, taylor, blocks, V_guess, dt, eta)[0];
        err = error_over(R_guess);
        perf = priv_ratio_max(err->bound, applied);
        bound = err->bound;
        priv_check_exploded(bound);
    }
    return *err;
}

NonlinearStep priv_lin_reach(const NonlinearSys &sys, const Zonotope &R, const Tensor &error,
                             const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    if (alg.linear_step != alg::LinearStep::cora) {
        // CORALean's region at tensor order 2, else CORA's loop with the step of CORALean's
        if (alg.error_bound == alg::ErrorBound::region && alg.tensor_order == 2)
            return aux_region_step(sys, R, U, alg);
        if (alg.tensor_order == 2) return aux_loop_step(sys, R, error, U, alg);
        return aux_lean_step(sys, R, error, U, alg);
    }
    const double dt = alg.time_step;
    const int eta = alg.taylor_terms;
    // the affine system and its step from R - x*
    const Linearized lin = sys.linearize(R, U, dt);
    AffineStep step = priv_affine_step(lin.sys, R - lin.x_star, lin.U, lin.u_const, dt, eta);
    const Zonotope &R_ti = step.time_interval;
    // the centered input set (CORA's params.U); without one, the input 0
    const std::optional<Zonotope> U0 = priv_centered_inputs(R.center(), sys.input_dim(), U);

    // the error: guessed, checked over the set it allows, and enlarged until it holds
    const AbstractionError err =
        priv_error_loop(lin, step.taylor, error, dt, eta, [&](const Zonotope &R_guess) {
            return priv_abstrerr_lin(sys, R_ti + R_guess, lin, U0, alg);
        });
    // the solution of the error moves both sets, which return to the coordinates of the state
    const Zonotope R_error =
        priv_particular_time_varying(lin.sys, step.taylor, step.blocks, err.set, dt, eta)[0];
    return {(R_ti + lin.x_star) + R_error, (step.time_point + lin.x_star) + R_error, err.bound};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
