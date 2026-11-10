// priv_tune_adaptive - the formulas that tune the step size and the order of the adaptive algorithm
//
// The upper exponential and logarithm by squaring and bisection, the gain of the abstraction error
// over a change of the step size (eq. (54)), the step size that minimizes the estimated size of
// the sets over the following steps (eqs. (57) and (58)), the gain that explains the error found
// at that step size (eq. (56)) and the choice of the tensor order (eqs. (46) and (47)). All of it
// is arithmetic on numbers, in the order CORALean's optDeltatFloat and estimateGainFloat compute.
//
// Syntax:   StepSize s = priv_opt_delta_t(alg, zeta_a, phi, r0, rabs, horizon);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  see the declarations in priv_adaptive.h
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_adapt_step.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The product of the gains of the first `scalings` scalings.
double aux_gain_product(double zeta_delta, double phi1, int scalings) {
    double product = 1;
    for (int j = 0; j < scalings; ++j) product *= priv_interp_gain(zeta_delta, phi1, j);
    return product;
}

/// Eq. (57): the size estimated for the set after the horizon, walked in 1 / zeta_delta^s steps.
double aux_size_estimate(const alg::LinearizationAdaptive &alg, double zeta_a, double phi,
                         double r0, double rabs, int s) {
    const double zd = alg.zeta_delta, zz = alg.zeta_z;
    const double lam = priv_pow(1 / zd, s);
    const int whole = int(std::floor(lam));
    const double b = lam - whole;
    const double base = 1 + (zz + zz);
    // the logarithms of the two bases are the same in every term
    const double log_base = priv_log_up(alg.exp_depth, alg.log_steps, base);
    const double log_a = priv_log_up(alg.exp_depth, alg.log_steps, zeta_a);
    const auto power = [&](double log_x, double y) {
        return priv_exp_up(alg.exp_depth, y * log_x);
    };
    // the sum over the whole steps, from the last term to the first as CORALean's sumUp
    double sum = 0;
    for (int j = whole - 1; j >= 0; --j) {
        const double e = b + double(j + 1);
        sum = power(log_base, e) * power(log_a, (e - 1) / lam) + sum;
    }
    const double zeta_az = sum + b * power(log_base, b);
    const double growth = (r0 * power(log_base, lam)) * zeta_a;
    return growth + ((rabs / lam) * zeta_az) * aux_gain_product(zd, phi, s);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The powers --------------------------------------------------------------------------------------

double priv_pow(double x, int k) {
    double power = 1;
    for (int i = 0; i < k; ++i) power *= x;
    return power;
}

// The exponential and the logarithm ---------------------------------------------------------------

double priv_exp_up(int depth, double a) {
    // the seed 1 + y + 2 y^2 at y = a / 2^depth, squared depth times
    const double y = std::ldexp(a, -depth);
    double u = (1 + y) + ((y * y) + (y * y));
    for (int k = 0; k < depth; ++k) u *= u;
    return u;
}

double priv_exp_down(int depth, double a) {
    const double y = std::ldexp(a, -depth);
    double u = std::max(0.0, 1 + y);
    for (int k = 0; k < depth; ++k) u = std::max(0.0, u * u);
    return u;
}

double priv_log_up(int depth, int steps, double a) {
    // bisection of [0, a]: the upper end moves where exp(mid) is at least a
    double lo = 0, hi = a;
    for (int k = 0; k < steps; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (a <= priv_exp_down(depth, mid))
            hi = mid;
        else
            lo = mid;
    }
    return hi;
}

// The gain ----------------------------------------------------------------------------------------

double priv_interp_gain(double zeta_delta, double phi1, int j) {
    return zeta_delta + (phi1 - zeta_delta) * priv_pow(zeta_delta, j);
}

StepSize priv_opt_delta_t(const alg::LinearizationAdaptive &alg, double zeta_a, double phi,
                          double r0, double rabs, double h) {
    // eq. (58): increase s while the estimate keeps falling, and stop at the first increase
    int s = 0;
    double cost = aux_size_estimate(alg, zeta_a, phi, r0, rabs, 0);
    for (int k = 0; k < alg.fuel; ++k) {
        const double next = aux_size_estimate(alg, zeta_a, phi, r0, rabs, s + 1);
        if (!(next <= cost)) break;
        cost = next;
        ++s;
    }
    return {h * priv_pow(alg.zeta_delta, s), s};
}

double priv_estimate_gain(double zeta_delta, double lam, double rabs1, double rabs_opt,
                          int scalings, int fuel) {
    // an error that is 0 at the horizon says nothing of the gain: the limit
    if (!(rabs1 > 0)) return zeta_delta;
    // eq. (56): bisection of the gain in [0, zeta_delta] whose product of gains meets the target
    const double target = (lam * rabs_opt) / rabs1;
    double lo = 0, hi = zeta_delta;
    for (int k = 0; k < fuel; ++k) {
        const double mid = 0.5 * (lo + hi);
        if (aux_gain_product(zeta_delta, mid, scalings) <= target)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

// The order ---------------------------------------------------------------------------------------

int priv_tune_order(double zeta_k, double phi, double rabs, double rabs_last, const Vec &d1,
                    const Vec &d2, int kappa) {
    // eq. (46): too close to re-tune
    if (std::abs((phi * rabs) / rabs_last) <= 1 - zeta_k) return kappa;
    // eq. (47): the second order wins too little, or it is worth its cost
    for (std::size_t i = 0; i < d1.size(); ++i)
        if (!(d1[i] <= 0 || zeta_k <= d2[i] / d1[i])) return 2;
    return 1;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
