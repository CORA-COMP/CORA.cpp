// priv_float_scalars - the rounded scalars and boxes of CORALean's float layer, at nearest
//
// Every rounded operation of CORALean's float layer is the machine's at nearest, in its order:
// factorialDown, powUp, taylorCoeffBounds, expRemainderBound, corrCoeffLow, normUpperBound and
// Interval.smul, and linComb of the zonotopes. Shared by the steps that follow CORALean
// (priv_float_step, priv_closed_form_step).
//
// See also: priv_float_step.h, priv_closed_form_step.h
//
// CORA.cpp only: the float definitions of CORALean's MatExp.
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The scalars -------------------------------------------------------------------------------------
// Every rounded operation of CORALean's float layer is the machine's at nearest, in its order.

/// k!, the product (k) * ((k-1) * ...) of CORALean's factorialDown.
inline double priv_float_factorial(int k) {
    double f = 1;
    for (int i = 1; i <= k; ++i) f = std::max(1.0, double(i) * f);
    return f;
}

/// a^k as a * (a * (... * 1)), CORALean's powUp.
inline double priv_float_power(int k, double a) {
    double p = 1;
    for (int i = 0; i < k; ++i) p = a * p;
    return p;
}

/// [r^k / k!] as CORALean's taylorCoeffBounds: the bounds of the quotient with 1/k! enclosed.
inline std::pair<double, double> priv_float_taylor_coeff(double r, int k) {
    const double inv = 1.0 / priv_float_factorial(k), p = priv_float_power(k, r);
    return {p * std::max(0.0, inv), p * inv};
}

/// The remainder bound of the series at order eta for the norm nrm: nrm^(eta+1) / (eta+1)! * 2.
inline double priv_float_remainder_bound(double nrm, int eta) {
    return (priv_float_power(eta + 1, nrm) / priv_float_factorial(eta + 1)) * (1.0 + 1.0);
}

/// -(q q) with q = (i - 1) / i: CORALean's lower bound of the correction coefficient.
inline double priv_float_corr_coeff_low(int i) {
    const double q = double(i - 1) / std::max(1.0, double(i));
    return -(q * q);
}

// The interval matrices ---------------------------------------------------------------------------

/// The largest row sum of the magnitudes of the entries, each sum nested to the right, floored at
/// 0 (CORALean's normUpperBound).
inline double priv_float_norm_upper_bound(const Interval &M) {
    const std::vector<double> lo = M.inf().data(), hi = M.sup().data();
    const std::size_t n = std::size_t(M.inf().size(-1)), rows = lo.size() / n;
    double nrm = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0;
        for (std::size_t j = n; j-- > 0;)
            sum = std::max(std::abs(lo[i * n + j]), std::abs(hi[i * n + j])) + sum;
        nrm = std::max(nrm, sum);
    }
    return nrm;
}

/// The box scaled by a number within [lo, hi], the hull of the corner products.
inline Interval priv_float_smul(const Interval &M, double lo, double hi) {
    const Tensor a = Tensor::like(M.inf(), {lo}, {1, 1}), b = Tensor::like(M.inf(), {hi}, {1, 1});
    return M.times(Interval(a, b));
}

/// The box with every entry [-w, w].
inline Interval priv_float_symmetric(const Tensor &like, double w) {
    return {like.zeros_like() - w, like.zeros_like() + w};
}

// The sets ----------------------------------------------------------------------------------------

/// The hull of the segments between matching points of Z1 and Z2 (CORALean's linComb): the paired
/// generators (G1 + G2) / 2, then (G1 - G2) / 2, the surplus of either copied as it is, and the
/// generator (c1 - c2) / 2. Equal counts of generators give linear_combination.
inline Zonotope priv_lin_comb(const Zonotope &Z1, const Zonotope &Z2) {
    const Tensor &G1 = Z1.generators(), &G2 = Z2.generators();
    const int64_t m1 = G1.size(-1), m2 = G2.size(-1), common = std::min(m1, m2);
    const Tensor a = G1.slice(-1, 0, common), b = G2.slice(-1, 0, common);
    std::vector<Tensor> parts{(a + b) / 2.0};
    if (m1 > common) parts.push_back(G1.slice(-1, common, m1));
    parts.push_back((a - b) / 2.0);
    if (m2 > common) parts.push_back(G2.slice(-1, common, m2));
    parts.push_back((Z1.center() - Z2.center()) / 2.0);
    return {(Z1.center() + Z2.center()) / 2.0, Tensor::concat(parts, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
