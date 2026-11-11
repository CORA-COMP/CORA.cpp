// priv_lin_reach_adaptive_pass - one linearization of the adaptive algorithm at a fixed step size
//
// The system is linearized at x* (the center of R moved half a step along the flow) and at the
// center of U. The step of the window is one linear step with the error as input: from the
// admissible error (a box, 0 at first) the set Rmax of the window is reached, the error over it
// is computed (priv_abstr_err_adaptive) and, if it exceeds the admissible one, enlarged by 1.1
// and the step repeated; it ends when the error is admissible, or gives up when the ratio to the
// admissible one does not fall after three passes. Then the sets are those with the error in the
// input, and the radius of its solution is what the tuning reads (CORALean's
// linReachAdaptivePassFloat).
//
// Syntax:   auto P = priv_lin_reach_adaptive_pass(sys, kappa, U, alg, time_step, R, error_adm);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  P - the sets and what the tuning reads of them; none if the loop gives up
// See also: priv_abstrerr_adaptive.cpp, LinearSys::one_step
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The linear step ---------------------------------------------------------------------------------

/// One step of x' = A (x - x*) + v, v in V, from R (oneStepFloat): the linear system's step from
/// R - x*, shifted back; the Taylor order is raised until it is admissible.
std::optional<std::pair<Zonotope, Zonotope>> aux_linear_step(const LinearSys &lin,
                                                             TaylorLinSys &taylor, int taylor_terms,
                                                             const Tensor &xs, const Zonotope &V,
                                                             const Zonotope &R, double dt) {
    // eta + 2 >= 2 ||A dt||, from the larger of eta and 2
    const double norm = lin.A().abs().sum(-1, true).max().item() * dt;
    int eta = std::max(taylor_terms, 2);
    while (double(eta + 2) < 2 * norm && eta < 4096) ++eta;
    if (!(double(eta + 2) >= 2 * norm)) return std::nullopt;
    // the constant input is the center of V, the time-varying one V without it
    const Tensor &v = V.center();
    const int64_t n = lin.dim();
    const Zonotope point(v, v.zeros_like({n, 0}));
    const BlockStep s =
        priv_one_step(lin, taylor, priv_one_block(n), R - xs, V - v, point, dt, eta);
    return std::make_pair(s.time_interval[0] + xs, s.time_point[0] + xs);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The pass ----------------------------------------------------------------------------------------

std::optional<AdaptivePass> priv_lin_reach_adaptive_pass(const NonlinearSys &sys, int kappa,
                                                         const std::optional<Zonotope> &U,
                                                         const alg::LinearizationAdaptive &alg,
                                                         double dt, const Zonotope &R,
                                                         const Vec &error_adm) {
    const Tensor &like = R.center();
    const int64_t n = sys.dim(), m = sys.input_dim();
    // x* = center + dt/2 f(center, u*), u* the center of U
    const Vec center = R.center().data(), us = m > 0 ? U->center().data() : Vec{};
    const auto point_u = [&](const Vec &u) {
        return m > 0 ? std::optional<Box>(Box{u, u}) : std::nullopt;
    };
    const Box fc = priv_enclosure(sys, like, Box{center, center}, point_u(us));
    Vec xs(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        xs[std::size_t(i)] = center[std::size_t(i)] +
                             (0.5 * dt) * (0.5 * (fc.lo[std::size_t(i)] + fc.hi[std::size_t(i)]));
    // the linearization and the input of the linear system without the error
    const Tensor xs_t = priv_column(like, xs);
    const Jacobians J =
        sys.jacobian(xs_t, m > 0 ? std::optional<Tensor>(priv_column(like, us)) : std::nullopt);
    Zonotope Vbase = priv_box_zonotope(like, priv_enclosure(sys, like, Box{xs, xs}, point_u(us)));
    if (m > 0) Vbase = Vbase + (*U - U->center()).linear_map(J.B);
    const LinearSys lin(J.A);
    TaylorLinSys taylor(J.A);

    // the error loop: the window with the admissible error as input, the error over it
    Vec admissible = error_adm;
    double previous = 0;
    std::optional<Zonotope> Rmax, Verror;
    Vec true_error;
    for (int pass = 0, counter = 1; pass < alg.fuel; ++pass, ++counter) {
        Box ebox{Vec(admissible.size()), admissible};
        for (std::size_t i = 0; i < admissible.size(); ++i) ebox.lo[i] = -admissible[i];
        const auto window = aux_linear_step(lin, taylor, alg.taylor_terms, xs_t,
                                            Vbase + priv_box_zonotope(like, ebox), R, dt);
        if (!window) return std::nullopt;
        Rmax = window->first;
        Verror = priv_abstr_err_adaptive(sys, kappa, J.A, J.B, xs, *Rmax, U, alg.zeta_z);
        true_error = priv_norm_mag(priv_zonotope_box(*Verror));
        // the largest ratio of the error to the admissible one
        double current = 0;
        bool within = true;
        for (std::size_t i = 0; i < admissible.size(); ++i) {
            current = std::max(current, true_error[i] <= 0 ? 0.0 : true_error[i] / admissible[i]);
            within = within && true_error[i] <= admissible[i];
        }
        if (within) break;
        if (2 < counter && !(current <= previous)) return std::nullopt;
        previous = current;
        for (std::size_t i = 0; i < admissible.size(); ++i) admissible[i] = 1.1 * true_error[i];
        if (pass + 1 == alg.fuel) return std::nullopt;
    }

    // the sets with the error in the input, and the radius of the solution of the error
    const auto S = aux_linear_step(lin, taylor, alg.taylor_terms, xs_t, Vbase + *Verror, R, dt);
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
