// con_poly_zono_testing - a known constrained polynomial zonotope and its point x0, for the tests
//
// S = {(1, 0) + a1 (1, 0.5) + a1 a2^2 (0, 1) + b (0.25, 0) | a1 - 0.5 a2 = 0.7}, its factor point
// a0 = (0.5, -0.4) that meets the row, the point x0 there (b = 0.3), and the test whether a point
// lies in the box of a set.
//
// Syntax:   ConPolyZono S = test::cpz::known();   x0 = test::cpz::point(S, kA0, {kBeta0});
// See also: test_con_poly_zono.cpp
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_poly_zono/con_poly_zono.h"
#include "testing.h"

#include <cmath>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test::cpz {

using namespace cora;

/// The factor point a0 = (0.5, -0.4) and the independent coefficient 0.3 of x0.
inline const std::vector<double> kA0 = {0.5, -0.4};
constexpr double kBeta0 = 0.3;

/// S = {(1, 0) + a1 (1, 0.5) + a1 a2^2 (0, 1) + b (0.25, 0) | a1 - 0.5 a2 = 0.7}: a0 meets the row.
inline ConPolyZono known() {
    return {test::vec({1.0, 0.0}),
            Tensor::from_data({1, 0.5, 0, 1}, {2, 2}),
            {{1, 0}, {1, 2}},
            Tensor::from_data({1, -0.5}, {1, 2}),
            test::vec({0.7}),
            {{1, 0}, {0, 1}},
            Tensor::from_data({0.25, 0}, {1, 2}),
            {1, 2}};
}

/// The point of the polynomial part at the factors a and the independent coefficients b.
inline std::vector<double> point(const ConPolyZono &S, const std::vector<double> &a,
                                 const std::vector<double> &b) {
    const int64_t n = S.dim();
    std::vector<double> x = S.c().data();
    const std::vector<double> g = S.generators().data(), gi = S.independent_generators().data();
    for (std::size_t j = 0; j < S.exponents().size(); ++j) {
        double mono = 1;
        for (std::size_t i = 0; i < a.size(); ++i) mono *= std::pow(a[i], S.exponents()[j][i]);
        for (int64_t r = 0; r < n; ++r) x[std::size_t(r)] += mono * g[j * std::size_t(n) + r];
    }
    for (std::size_t j = 0; j < b.size(); ++j)
        for (int64_t r = 0; r < n; ++r) x[std::size_t(r)] += b[j] * gi[j * std::size_t(n) + r];
    return x;
}

/// Whether the point x lies in the box of S, within 1e-9.
inline bool in_box(const ContSet &S, const std::vector<double> &x) {
    const Interval I = S.interval();
    const std::vector<double> lo = I.inf().data(), hi = I.sup().data();
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] < lo[i] - 1e-9 || x[i] > hi[i] + 1e-9) return false;
    return true;
}

} // namespace test::cpz

// ---------------------------------------  END OF CODE  ---------------------------------------- //
