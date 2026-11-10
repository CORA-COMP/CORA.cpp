// priv_tune_adaptive - the formulas that tune the step size of the adaptive algorithm
//
// The gain of the abstraction error over a change of the step size (CORA's aux_varphitotal), the
// step size that minimizes the estimated width of the sets over the following steps
// (aux_optimaldeltat, eq. (49) of [1]) and the gain that explains the error measured at a smaller
// step size (aux_varphiest, by bisection), with the small vector helpers they and the step share.
// All of it is arithmetic on numbers: nothing here changes the options.
//
// Syntax:   double dt = priv_optimal_delta_t(Rstart, Rerror, horizon, varphi, zeta_p, run);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  see the declarations in priv_adaptive.h
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_lin_reach_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace


// ===========================================  MAIN  =========================================== //

// The helpers -------------------------------------------------------------------------------------

/// The sum of the lengths of the generators per coordinate.
std::vector<double> priv_row_sums(const Zonotope &Z) {
    return Z.generators().abs().sum(-1, true).data();
}

/// The Euclidean norm of the entries of the vector.
double priv_vector_norm(const std::vector<double> &v) {
    // the root of the sum of squares
    double sum = 0;
    for (const double x : v) sum += x * x;
    return std::sqrt(sum);
}

/// The entries of v at the places where orders equals the minimal order.
std::vector<double> priv_at_min_order(const std::vector<double> &v, const AdaptiveRun &run) {
    std::vector<double> out;
    for (std::size_t j = 0; j < v.size(); ++j)
        if (run.orders[j] == run.min_order) out.push_back(v[j]);
    return out;
}

/// The step size gain zeta_phi of the minimal order (CORA's zetaphi(minorder + 1)).
double priv_zeta_phi(const AdaptiveRun &run) {
    if (run.min_order < 0 || std::size_t(run.min_order) >= run.alg.zeta_phi.size())
        CORA_FAIL(Error::not_supported,
                  "reach: the abstraction error of order " + std::to_string(run.min_order) +
                      " is not tuned (zeta_phi has entries for the orders 0 to 2)");
    return run.alg.zeta_phi[std::size_t(run.min_order)];
}

// The formulas -----------------------------------------------------------------------------------

/// The product of the gains over the scalings from step size `prev` to `curr` (CORA's
/// aux_varphitotal): each scaling by mu has a gain between varphi and the limit.
double priv_varphitotal(double varphi, double varphi_lim, double prev, double curr, double mu) {
    const double scalings = std::log(prev / curr) / std::log(mu);
    const double eps = machine_eps();
    std::vector<double> phis{varphi};
    // one gain per scaling of the step size, the last one blended towards 1
    if (scalings > 0) {
        const int low = int(std::floor(scalings + 10 * eps)), high = low + 1;
        for (int i = 1; i <= low; ++i)
            phis.push_back(varphi +
                           (varphi_lim - varphi) * (curr - std::pow(mu, i) * curr) / (curr - 0));
        phis.back() = phis.back() + (1 - phis.back()) * (prev - curr * std::pow(mu, high)) /
                                        (curr * (std::pow(mu, low) - std::pow(mu, high)));
    } else {
        const int low = int(std::ceil(scalings - 10 * eps)), high = low - 1;
        for (int i = -1; i >= low; --i)
            phis.push_back(varphi +
                           (varphi_lim - varphi) * (curr - std::pow(mu, i) * curr) / (curr - 0));
        phis.back() = phis.back() + (1 - phis.back()) * (prev - curr * std::pow(mu, high)) /
                                        (curr * (std::pow(mu, low) - std::pow(mu, high)));
        for (double &x : phis) x = 1 / x;
    }
    // the gains multiply (the reciprocals, when the step size grows)
    double prod = 1;
    for (const double x : phis) prod *= x;
    return prod;
}

/// The step size below the horizon that minimizes the estimated width of the sets over the
/// following steps (CORA's aux_optimaldeltat, eq. (49) of [1]).
double priv_optimal_delta_t(const Zonotope &Rt, const Zonotope &Rerr, double deltat,
                            double varphimin, double zeta_p, const AdaptiveRun &run) {
    const double mu = run.alg.decr_factor;
    const double zeta_z = run.alg.zeta_z * 0.5; // a better representation of the actual reduction
    const int kprimemax = int(std::ceil(-std::log(100.0) / std::log(mu)));
    const double r_start = priv_vector_norm(priv_row_sums(Rt)),
                 rerr1 = priv_vector_norm(priv_row_sums(Rerr));
    const double varphi_h = mu - varphimin;
    // the candidates: the horizon scaled down by mu, k = mu^-k' times
    std::vector<double> deltats, k, floork;
    for (int kp = 0; kp <= kprimemax; ++kp) {
        k.push_back(std::pow(mu, -kp));
        deltats.push_back(deltat * std::pow(mu, kp));
        floork.push_back(std::floor(k.back()));
    }
    // the objective over the candidates
    double varphiprod = 1;
    int best = -1;
    double best_value = kInf;
    for (int kp = 0; kp <= kprimemax; ++kp) {
        const auto q = std::size_t(kp);
        varphiprod *= (varphimin + (deltats[0] - deltats[q]) / deltats[0] * varphi_h) / mu;
        double all_but_last = 0;
        for (int j = 1; j <= int(floork[q]); ++j)
            all_but_last += std::pow(1 + 2 * zeta_z, k[q] + 1 - j) * std::pow(zeta_p, 1 - j / k[q]);
        const double objective =
            r_start * std::pow(1 + 2 * zeta_z, k[q]) * zeta_p +
            rerr1 / k[q] * varphiprod *
                (all_but_last + std::pow(1 + zeta_z, k[q] - kp) * (k[q] - floork[q]));
        if (!std::isnan(objective) && (best < 0 || objective < best_value)) {
            best = kp;
            best_value = objective;
        }
    }
    return deltats[std::size_t(std::max(best, 0))];
}

/// The gain of the abstraction error that makes the error of a step of size deltat equal to the
/// one measured at the horizon, by bisection (CORA's aux_varphiest).
double priv_varphi_est(double horizon, double deltat, const Zonotope &Rerr_h,
                       const Zonotope &Rerr_deltat, const AdaptiveRun &run) {
    const double mu = run.alg.decr_factor;
    const double rerr1 = priv_vector_norm(priv_at_min_order(priv_row_sums(Rerr_h), run));
    const double rerrk = priv_vector_norm(priv_at_min_order(priv_row_sums(Rerr_deltat), run));
    if (!(rerr1 > rerrk))
        CORA_FAIL(Error::report_to_dev, "reach: check the abstraction errors of the tuning");
    // bisection for the gain whose total equals the ratio of the errors
    const double rhs = rerrk / rerr1;
    const double varphi_lim = std::pow(mu, run.min_order + 1);
    double low = 0, up = std::pow(mu, run.min_order + 1), previous = 0;
    for (int count = 1; count <= 10000; ++count) {
        const double varphi = low + 0.5 * (up - low);
        const double residual = priv_varphitotal(varphi, varphi_lim, deltat, horizon, mu) - rhs;
        if (residual < 0)
            low = varphi;
        else
            up = varphi;
        if (std::abs(residual) < numerics::kGainResidual ||
            (count > 1 && std::abs(varphi - previous) < numerics::kGainStep))
            return varphi;
        previous = varphi;
    }
    CORA_FAIL(Error::special_error, "Bug in varphi estimation... report to devs");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
