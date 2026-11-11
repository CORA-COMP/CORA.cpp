// priv_lin_reach_adaptive_pass - one linearization of the adaptive algorithm at a fixed step size
//
// The system is linearized at x* (the center of R moved half a step along the flow) and at the
// center of U. The step of the window is one linear step (priv_float_one_step) with the error as
// input: from the admissible error (a box, 0 at first) the set Rmax of the window is reached, the
// error over it is computed (priv_abstr_err_adaptive) and, if it exceeds the admissible one,
// enlarged by 1.1 and the step repeated; it ends when the error is admissible, or gives up when the
// ratio to the admissible one does not fall after three passes. Then the sets are those with the
// error in the input, and the radius of its solution is what the tuning reads (CORALean's
// linReachAdaptivePassFloat).
//
// Syntax:   auto P = priv_lin_reach_adaptive_pass(sys, kappa, U, alg, time_step, R, error_adm);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  P - the sets and what the tuning reads of them; none if the loop gives up
// See also: priv_abstrerr_adaptive.cpp, priv_float_step.h
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_step.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ===========================================  MAIN  =========================================== //

// The pass ----------------------------------------------------------------------------------------

std::optional<AdaptivePass> priv_lin_reach_adaptive_pass(const NonlinearSys &sys, int kappa,
                                                         const std::optional<Zonotope> &U,
                                                         const alg::LinearizationAdaptive &alg,
                                                         double dt, const Zonotope &R,
                                                         const Vec &error_adm) {
    const Tensor &like = R.center();
    const int64_t n = sys.dim();
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, U, like, dt);
    const Vec &xs = L.xs;
    const Tensor &xs_t = L.xs_t;
    const Jacobians &J = L.J;
    const Zonotope &Vbase = L.Vbase;
    const LinearSys lin(J.A);

    // the error loop: the window with the admissible error as input, the error over it
    std::optional<Zonotope> Rmax, Verror;
    bool failed = false;
    const std::optional<Vec> found =
        priv_error_loop_adaptive(error_adm, alg.fuel, [&](const Vec &adm) {
            Box ebox{Vec(adm.size()), adm};
            for (std::size_t i = 0; i < adm.size(); ++i) ebox.lo[i] = -adm[i];
            const auto window =
                priv_float_one_step(J.A, xs_t, Vbase + priv_box_zonotope(like, ebox), R, dt,
                                    alg.taylor_terms, alg.zonotope_order);
            if (!window) {
                failed = true;
                return Vec(adm.size(), 0.0);
            }
            Rmax = window->first;
            Verror = priv_abstr_err_adaptive(sys, kappa, J.A, J.B, xs, *Rmax, U, alg.zeta_z);
            return priv_norm_mag(priv_zonotope_box(*Verror));
        });
    if (failed || !found) return std::nullopt;
    const Vec &true_error = *found;

    // the sets with the error in the input, and the radius of the solution of the error
    const auto S = priv_float_one_step(J.A, xs_t, Vbase + *Verror, R, dt, alg.taylor_terms,
                                       alg.zonotope_order);
    if (!S) return std::nullopt;
    const Zonotope Rerror =
        lin.error_solution_adaptive(*Verror, std::nullopt, dt, 0.0, alg.taylor_terms);
    const Vec error_radius = priv_row_sums(*Verror);
    std::vector<char> gzero(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < gzero.size(); ++i) gzero[i] = error_radius[i] <= 0;
    return AdaptivePass{S->first, S->second, J.A, xs, *Rmax, true_error, priv_row_sums(Rerror),
                        gzero};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
