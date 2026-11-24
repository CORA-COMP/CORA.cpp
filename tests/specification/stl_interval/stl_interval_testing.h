// stl_interval_testing - what the tests of the STL intervals share
//
// CORA.cpp only: test helpers
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/rng.h"
#include "cora/specification/stl_interval.h"

#include <cmath>
#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace stl_test {

inline const double kInf = std::numeric_limits<double>::infinity();

/// Whether the time t is in the interval, by the definition of the ends.
inline bool member(const cora::StlInterval &I, double t) {
    if (I.is_empty_object()) return false;
    return (I.lower() < t || (I.left_closed() && I.lower() == t)) &&
           (t < I.upper() || (I.right_closed() && I.upper() == t));
}

/// The times 0, 1/4, ..., 9: sums and differences of them are exact.
inline std::vector<double> grid() {
    std::vector<double> times;
    for (int k = 0; k <= 36; ++k) times.push_back(k / 4.0);
    return times;
}

/// Empty, point, unbounded and bounded intervals with all open and closed ends, and random ones
/// (drawn from rng).
inline std::vector<cora::StlInterval> samples(cora::Rng &rng) {
    std::vector<cora::StlInterval> out{cora::StlInterval(), cora::StlInterval(1),
                                       cora::StlInterval(0, kInf)};
    for (const double lb : {0.0, 1.0, 2.0})
        for (const double ub : {2.0, 3.0, 4.0})
            for (const bool lc : {true, false})
                for (const bool rc : {true, false}) out.emplace_back(lb, ub, lc, rc);
    out.emplace_back(1.0, kInf, true, true);
    // random intervals with bounds on the grid of quarters
    std::vector<double> u(4 * 12);
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    for (std::size_t k = 0; k < 12; ++k) {
        const double a = std::floor(u[4 * k] * 16) / 4, w = std::floor(u[4 * k + 1] * 12) / 4;
        out.emplace_back(a, a + w, u[4 * k + 2] < 0.5, u[4 * k + 3] < 0.5);
    }
    return out;
}

} // namespace stl_test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
