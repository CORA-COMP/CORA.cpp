// level_set_testing - what the tests of the level set share
//
// Syntax:   Expr f = test::circle(1, 1, 0.5);   Interval I = test::box({0, 0}, {2, 2});
// Sets:     the circle (x - cx)^2 + (y - cy)^2 - r^2 of two states, boxes of host numbers
// See also: testing.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/level_set/level_set.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The expression (x - cx)^2 + (y - cy)^2 - r^2 of the states 0 and 1.
inline cora::Expr circle(double cx, double cy, double r) {
    const cora::Expr x = cora::Expr::var(0), y = cora::Expr::var(1);
    return cora::pow(x - cx, 2) + cora::pow(y - cy, 2) - r * r;
}

/// The box [lo, hi] of host numbers.
inline cora::Interval box(const std::vector<double> &lo, const std::vector<double> &hi) {
    return {column(lo), column(hi)};
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
