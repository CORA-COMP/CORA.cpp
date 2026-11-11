// priv_dae_lin_reach - one step of a differential-algebraic system by linearization, as linReach
//
// The step linearizes at (x*, y*, u*) (linearize), solves the affine system from R - x* by one step
// of LinearSys (oneStep), and adds the solution of the linearization error L. L is not known before
// the step: its bounds on f and on g are guessed as 1.1 times those of the step before (0 at the
// first), the set of the step is enlarged by the solution of the guess, and the error over that set
// is computed (priv_dae_lin_error). While a bound exceeds its guess somewhere, the guess becomes
// the bound and the loop repeats; then the solution of the error is added to both sets. MATLAB's
// max skips NaN, so an error of 0 over a guess of 0 does not count. An error beyond maxError would
// make CORA split the set; its call reads the undefined variable `iter` and aborts, so the split
// is not ported and the step fails with not_supported.
//
// Syntax:   DaeStep s = priv_dae_lin_reach(sys, R, y0, error_x, error_y, U, u_trans, dec, alg);
// Inputs:   R - the set at the start, y0 - a guess of y at its center, error_x, error_y - the
//           bounds of the step before, U - the input set, u_trans - the input, dec - or none
// Outputs:  s - the sets over the step and at its end, y over the step, the bounds of this step
// See also: priv_dae_lin_error, NonlinDASys::linearize, LinearSys::one_step
//
// CORA: contDynamics/@nonlinDASys/linReach.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlin_das_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The bounds --------------------------------------------------------------------------------------

/// MATLAB's max(a, b) of two numbers, either of which may be NaN: NaN only if both are.
double aux_max(double a, double b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return std::max(a, b);
}

/// The bound per entry as a column: `limit` if given, else infinity (CORA's default Inf).
Tensor aux_limit(const std::vector<double> &limit, const Tensor &like, int64_t n) {
    if (limit.empty()) return like.full_like({n, 1}, std::numeric_limits<double>::infinity());
    return Tensor::like(like, limit, {n, 1});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The step ----------------------------------------------------------------------------------------

DaeStep priv_dae_lin_reach(const NonlinDASys &sys, const Zonotope &R, const Tensor &y0,
                           const Tensor &error_x, const Tensor &error_y,
                           const std::optional<Zonotope> &U, const Tensor &u_trans,
                           const std::optional<DaeDecomposition> &decomposition,
                           const alg::DaeLinearization &alg) {
    const double dt = alg.time_step;
    const int eta = alg.taylor_terms;
    const int64_t n = sys.dim(), p = sys.constraint_dim(), m = sys.input_dim();
    // the affine system and its step from R - x*
    const DaeLinearized lin = sys.linearize(R.center(), y0, U, u_trans, dt);
    AffineStep step = priv_affine_step(lin.sys, R - lin.x_star, lin.U, lin.u_const, dt, eta);
    const Zonotope &R_ti = step.time_interval;
    // the centered input set (CORA's params.U); without one, the input 0
    const std::optional<Zonotope> U0 = priv_centered_inputs(R.center(), m, U);
    const Tensor max_x = aux_limit(alg.max_error_x, R.center(), n);
    const Tensor max_y = aux_limit(alg.max_error_y, R.center(), p);

    // the error: guessed, checked over the set it allows, and enlarged until it holds
    Tensor bound_x = error_x, bound_y = error_y, applied_x = error_x, applied_y = error_y;
    double perf_x_cur = std::numeric_limits<double>::infinity(), perf_y_cur = perf_x_cur;
    double perf = 0;
    std::optional<DaeError> err;
    while ((perf_x_cur > 1 || perf_y_cur > 1) && perf <= 1) {
        // a bound that was exceeded becomes 1.1 times the bound found
        if (perf_x_cur > 1) applied_x = bound_x * 1.1;
        if (perf_y_cur > 1) applied_y = bound_y * 1.1;
        const Zonotope V_x(applied_x.zeros_like(), applied_x.diag());
        const Zonotope V_y(applied_y.zeros_like(), applied_y.diag());
        // the set the guess allows, and the error over it
        const Zonotope R_all = priv_particular_time_varying(
            lin.sys, step.taylor, step.blocks, V_x + V_y.linear_map(lin.CF_inv), dt, eta)[0];
        err = priv_dae_lin_error(sys, lin, R_ti + R_all, V_y, U0, decomposition, alg);
        // the bounds against the guesses, and against the largest errors allowed
        perf_x_cur = priv_ratio_max(err->bound_x, applied_x);
        perf_y_cur = priv_ratio_max(err->bound_y, applied_y);
        perf = aux_max(priv_ratio_max(err->bound_x, max_x), priv_ratio_max(err->bound_y, max_y));
        bound_x = err->bound_x;
        bound_y = err->bound_y;
        priv_check_exploded(err->bound);
    }
    if (perf > 1)
        CORA_FAIL(Error::not_supported,
                  "reach: the linearization error exceeds max_error_x or max_error_y, where CORA "
                  "splits the set; its split reads an undefined variable and aborts, and is not "
                  "ported");
    // the solution of the error moves both sets, which return to the coordinates of the state
    const Zonotope R_error =
        priv_particular_time_varying(lin.sys, step.taylor, step.blocks, err->set, dt, eta)[0];
    return {(R_ti + lin.x_star) + R_error, (step.time_point + lin.x_star) + R_error, err->algebraic,
            bound_x, bound_y};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
