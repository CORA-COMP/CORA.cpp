// test_scalar_interval - the host interval of doubles that the scalar interval arithmetics share
//
// The two corner products (IEEE, and 0 * Inf taken as 0), the hull's handling of equal and NaN
// corners, the square, the meet and the widening.
//
// CORA.cpp only: a helper of the level-set and Taylor model code
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#include "global/scalar_interval.h"
#include "testing.h"

#include <cmath>
#include <limits>

using cora::detail::ScalarInterval;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool aux_is(const ScalarInterval &a, double lo, double hi) { return a.lo == lo && a.hi == hi; }

/// Sums, negation and the two products of ordinary intervals.
void arithmetic() {
    const ScalarInterval a{-1, 2}, b{3, 4};
    check(aux_is(a + b, 2, 6) && aux_is(a - b, -5, -1) && aux_is(-a, -2, 1), "sums and negation");
    check(aux_is(cora::detail::times(a, b), -4, 8), "times");
    check(aux_is(cora::detail::times_zero_inf(a, b), -4, 8), "times_zero_inf on finite bounds");
    check(aux_is(cora::detail::square(a), 0, 4) && aux_is(cora::detail::square(-b), 9, 16),
          "square");
    check(cora::detail::meet(a, b).empty() && aux_is(cora::detail::meet(a, {0, 5}), 0, 2), "meet");
    check(a.has_zero() && !b.has_zero() && a.center() == 0.5 && a.rad() == 1.5, "the measures");
}

/// 0 * Inf: NaN in times (the hull keeps a NaN first corner), 0 in times_zero_inf.
void zero_times_infinity() {
    const ScalarInterval zero{0, 0}, whole = ScalarInterval::whole();
    const ScalarInterval ieee = cora::detail::times(zero, whole);
    check(std::isnan(ieee.lo) && std::isnan(ieee.hi), "times: 0 * Inf is NaN");
    check(aux_is(cora::detail::times_zero_inf(zero, whole), 0, 0), "times_zero_inf: 0 * Inf is 0");
    const ScalarInterval w = cora::detail::widen({1, 2});
    check(w.lo < 1 && w.hi > 2 && w.lo == std::nextafter(1.0, -kInf), "widen by one ulp");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    arithmetic();
    zero_times_infinity();
    return test::finish("scalar interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
