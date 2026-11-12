// priv_lin_reach_poly_adaptive_pass - one linearization of poly-adaptive at a fixed step size
//
// Linearizes at x* = c + dt/2 f(c, u*), c the center of the start set R, and solves the affine
// system from Rdelta = R - x* in a polynomial zonotope. The error loop guesses the admissible
// error, takes the window over the step with the guess as an input (RallError), the differences
// Rdiff of that window and the region Rmax they span, and computes the dynamic error VerrorDyn over
// Rmax (priv_abstr_err_poly_adaptive); the true error is the largest magnitude of VerrorDyn and the
// static error together. It stops where that is within the guess, and else enlarges the guess by
// 1.1 times the true error, giving up if the ratio does not shrink after two rounds. The sets then
// have the dynamic error as an input of the window (Rti with the static error in it, Rtp with its
// dependent part through exact_plus), and the tuning reads the radius of the solutions of the
// errors (CORALean's linReachPolyAdaptivePassFloat).
//
// Syntax:   auto P = priv_lin_reach_poly_adaptive_pass(sys, U, alg, dt, R, error_adm);
// Inputs:   R - the start set
//           error_adm - the admissible error (n)
//           dt - the step size
// Outputs:  P - the sets and the true error, none if the loop or a linear step fails
// See also: priv_lin_reach_adaptive_pass.cpp, priv_poly_window.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<int> priv_reach_order(const Tensor &A, double dt, int eta) {
    // eta + 2 >= 2 ||A dt||, from the larger of eta and 2
    const double norm = A.abs().sum(-1, true).max().item() * dt;
    eta = std::max(eta, 2);
    while (double(eta + 2) < 2 * norm && eta < 4096) ++eta;
    if (!(double(eta + 2) >= 2 * norm)) return std::nullopt;
    return eta;
}

std::optional<PolyAdaptivePass>
priv_lin_reach_poly_adaptive_pass(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                                  const alg::PolynomializationAdaptive &alg, double dt,
                                  const PolyZonotope &R, const Vec &error_adm) {
    // a step of positive width, linearized at the center of the start set
    if (!(dt > 0)) return std::nullopt;
    const Tensor &like = R.center();
    const int64_t n = sys.dim();
    const AdaptiveLinearization L = priv_linearize_adaptive(sys, U, like, dt);
    const Vec &xs = L.xs;
    const Tensor &xs_t = L.xs_t;
    const Jacobians &J = L.J;
    const Zonotope &Vbase = L.Vbase;
    const LinearSys lin(J.A);
    TaylorLinSys taylor(J.A);
    const std::optional<int> eta = priv_reach_order(J.A, dt, alg.taylor_terms);
    if (!eta) return std::nullopt;

    // the start set relative to x*, and the static error over it
    const PolyZonotope Rdelta = R.plus(xs_t * -1.0);
    const PolyStatic S = priv_precomp_stat_error_adaptive(sys, xs, U, alg.zeta_z, Rdelta);
    const Zonotope Vsz = S.Vstat.zonotope();

    // the error loop: the window with the admissible error as input, the error over its region
    std::optional<Zonotope> Verror;
    const std::optional<Vec> found =
        priv_error_loop_adaptive(error_adm, alg.fuel, [&](const Vec &adm) {
            Box ebox{Vec(adm.size()), adm};
            for (std::size_t i = 0; i < adm.size(); ++i) ebox.lo[i] = -adm[i];
            const auto window = priv_poly_window(
                lin, taylor, dt, *eta, Vbase + priv_box_zonotope(like, ebox), Rdelta, alg.zeta_z);
            const Zonotope Rmax = (Rdelta.zonotope() + window.second.zonotope()) + xs_t;
            Verror =
                priv_abstr_err_poly_adaptive(sys, xs, Rmax, S.Zdelta, window.second, U, alg.zeta_z);
            return priv_norm_mag(priv_zonotope_box(*Verror + Vsz));
        });
    if (!found) return std::nullopt;
    const Vec &true_error = *found;

    // the sets: the dynamic and the static error in the window, the dynamic error in the end set
    const Zonotope Vdyn = *Verror;
    const PolyZonotope Rti =
        priv_poly_window(lin, taylor, dt, *eta, (Vbase + Vdyn) + Vsz, Rdelta, alg.zeta_z).first;
    const PolyZonotope Rtp =
        priv_poly_point(lin, taylor, dt, *eta, Vbase + Vdyn, Rdelta, S.Vstat, alg.zeta_z);

    // the radius of the solutions of the errors (independent generators of both)
    const Zonotope Rdyn = lin.error_solution_adaptive(Vdyn, std::nullopt, dt, 0.0, eta);
    const PolyZonotope Rstat = S.Vstat.linear_map(priv_adaptive_integral(lin, taylor, dt, *eta));
    const Tensor GI = Tensor::concat({Rdyn.generators(), Rstat.independent_generators()}, -1);
    const int64_t q = GI.shape()[1];
    const Vec gi = GI.data();
    Vec abstrerr(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        Vec row(static_cast<std::size_t>(q));
        for (int64_t j = 0; j < q; ++j) row[std::size_t(j)] = std::abs(gi[std::size_t(i * q + j)]);
        abstrerr[std::size_t(i)] = priv_sum_up(row);
    }
    const Vec error_radius = priv_row_sums(Vdyn);
    std::vector<char> gzero(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < gzero.size(); ++i) gzero[i] = error_radius[i] <= 0;
    return PolyAdaptivePass{Rti.plus(xs_t), Rtp.plus(xs_t), J.A, xs, true_error, abstrerr, gzero};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
