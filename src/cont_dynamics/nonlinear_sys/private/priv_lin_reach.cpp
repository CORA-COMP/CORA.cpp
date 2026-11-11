// priv_lin_reach - one step of a nonlinear system by conservative linearization, as CORA's linReach
//
// The step linearizes at (x*, u*) (linearize), solves the affine system from R - x* by one step
// of LinearSys (oneStep), and adds the solution of the linearization error L. L is not known
// before the step: it is guessed as 1.1 times the error of the step before (0 at the first), the
// set of the step is enlarged by the solution of the guess, and the error over that set is
// computed (priv_abstrerr_lin). While the error exceeds the guess somewhere, the guess becomes the
// error and the loop repeats; then the solution of the error is added to both sets. MATLAB's max
// skips NaN, so an error of 0 over a guess of 0 does not count, and all-NaN ends the loop.
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

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


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
