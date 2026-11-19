// priv_krylov_rounding - CORALean's float primitives at the dtype nearest, for the Krylov sources
//
// `absolute`, `maximum`, `minimum`, `maximumOver`, `half` and the sum of CORALean's definitions as
// they evaluate at nearest: the sum runs from the last term, nothing is fused (the including file
// is compiled with -ffp-contract=off). Shared by the Krylov and the verify sources.
//
// Syntax:   double a = lean_absolute(x);   double s = lean_sum(n, [&](int64_t i) { return f(i); });
// See also: priv_krylov.h
//
// CORA.cpp only: the rounding of CORALean's definitions has no CORA counterpart.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// CORALean's `absolute`: `maximum a (-a)`, so that |0| is -0.
inline double lean_absolute(double a) { return a <= -a ? -a : a; }

/// CORALean's `maximum` (NaN stays).
inline double lean_maximum(double a, double b) { return a <= b ? b : (b < a ? a : a + b); }

/// CORALean's `minimum` (NaN stays).
inline double lean_minimum(double a, double b) { return a <= b ? a : (b < a ? b : a + b); }

/// `maximumOver f` of n values: nested to the right from the last; 0 for none.
template <class F>
double lean_max_over(int64_t n, const F &f) {
    if (n == 0) return 0.0;
    double acc = f(n - 1);
    for (int64_t i = n - 2; i >= 0; --i) acc = lean_maximum(f(i), acc);
    return acc;
}

/// CORALean's `half`: exact, NaN where the result is not normal.
inline double lean_half(double a) {
    const double r = a * 0.5;
    if (r == 0.0 && a != 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(r) && (r == 0.0 || std::abs(r) >= DBL_MIN)))
        return std::numeric_limits<double>::quiet_NaN();
    return r;
}

/// `sumUp f` of n terms: nested to the right, so summed from the last term, from zero.
template <class F>
double lean_sum(int64_t n, F f) {
    double acc = 0.0;
    for (int64_t i = n; i-- > 0;) {
        const double term = f(i);
        acc = term + acc;
    }
    return acc;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
