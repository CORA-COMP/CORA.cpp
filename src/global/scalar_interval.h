// scalar_interval - one closed interval of doubles on the host, for the scalar interval arithmetics
//
// The level-set contractor and quadratic evaluation and the Taylor model remainder compute with
// single intervals of numbers, where an Interval of tensors would cost a tensor per operation.
// They share this type. The corner products come in the two conventions the callers need: times
// takes IEEE products (0 * Inf is NaN), times_zero_inf takes a corner 0 * Inf as 0. A hull keeps
// the first of equal corners and a NaN in the first corner, as std::min and std::max do.
//
// Syntax:   ScalarInterval a{lo, hi};   a + b;   a - b;   -a;   times(a, b);   times_zero_inf(a, b)
// See also: cora/cont_set/interval/interval.h (the Interval of tensors)
//
// CORA.cpp only: CORA computes these with its interval class; here a host type serves the callers
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::detail {

/// The interval [lo, hi] of doubles; empty if lo > hi.
struct ScalarInterval {
    double lo = 0, hi = 0;

    bool empty() const { return lo > hi; }
    bool has_zero() const { return lo <= 0 && hi >= 0; }
    double center() const { return 0.5 * (lo + hi); }
    double rad() const { return 0.5 * (hi - lo); }

    /// The whole line (-Inf, Inf).
    static ScalarInterval whole() {
        constexpr double kInf = std::numeric_limits<double>::infinity();
        return {-kInf, kInf};
    }
};

inline ScalarInterval operator+(const ScalarInterval &a, const ScalarInterval &b) {
    return {a.lo + b.lo, a.hi + b.hi};
}
inline ScalarInterval operator-(const ScalarInterval &a, const ScalarInterval &b) {
    return {a.lo - b.hi, a.hi - b.lo};
}
inline ScalarInterval operator-(const ScalarInterval &a) { return {-a.hi, -a.lo}; }

/// The intersection of two intervals (empty if they do not meet).
inline ScalarInterval meet(const ScalarInterval &a, const ScalarInterval &b) {
    return {std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
}

/// The hull of the four corners, in the order (lo lo, lo hi, hi lo, hi hi).
inline ScalarInterval hull_of_corners(const std::array<double, 4> &v) {
    double lo = v[0], hi = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        lo = std::min(lo, v[i]);
        hi = std::max(hi, v[i]);
    }
    return {lo, hi};
}

/// The product with IEEE corner products: 0 * Inf is NaN.
inline ScalarInterval times(const ScalarInterval &a, const ScalarInterval &b) {
    return hull_of_corners({a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi});
}

/// The product with a corner 0 * Inf taken as 0 (an exact 0 factor), as CORA's interval mtimes.
inline ScalarInterval times_zero_inf(const ScalarInterval &a, const ScalarInterval &b) {
    const auto corner = [](double x, double y) { return x == 0.0 || y == 0.0 ? 0.0 : x * y; };
    return hull_of_corners(
        {corner(a.lo, b.lo), corner(a.lo, b.hi), corner(a.hi, b.lo), corner(a.hi, b.hi)});
}

/// The square: nonnegative, 0 at the lower end if the interval holds 0.
inline ScalarInterval square(const ScalarInterval &a) {
    const double lo = a.lo * a.lo, hi = a.hi * a.hi;
    if (a.lo >= 0) return {lo, hi};
    if (a.hi <= 0) return {hi, lo};
    return {0.0, std::max(lo, hi)};
}

/// Widened by one ulp on both sides, against the rounding of a root or a logarithm.
inline ScalarInterval widen(const ScalarInterval &a) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    return {std::nextafter(a.lo, -kInf), std::nextafter(a.hi, kInf)};
}

} // namespace cora::detail

// ---------------------------------------  END OF CODE  ---------------------------------------- //
