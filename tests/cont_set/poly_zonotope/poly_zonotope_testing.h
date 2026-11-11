// poly_zonotope_testing - what the tests of the polynomial zonotope share
//
// Syntax:   PolyZonotope P = test::random_poly(rng, n);   bool in = test::inside(P, points);
// Sets:     the center and the generators are standard normal; two factors with exponents 0..3
// See also: testing.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// A tensor (rows, cols) of standard normal entries.
inline cora::Tensor normal(cora::Rng &rng, int64_t rows, int64_t cols) {
    std::vector<double> v(std::size_t(rows * cols));
    rng.normal(v.data(), v.size(), 1.0);
    return cora::Tensor::from_data(v, {rows, cols});
}

/// A random polynomial zonotope of n dimensions with 4 dependent and 2 independent generators on 2
/// factors.
inline cora::PolyZonotope random_poly(cora::Rng &rng, int64_t n) {
    const cora::Exponents E = {{1, 0}, {0, 1}, {2, 1}, {1, 2}};
    return {normal(rng, n, 1), normal(rng, n, 4), normal(rng, n, 2), E, {1, 2}};
}

/// Whether every column of the points lies in the zonotope up to a small tolerance.
inline bool inside(const cora::Zonotope &Z, const cora::Tensor &points) {
    return Z.contains(points, "exact", 1e-8);
}

/// The points of a set: N random ones.
inline cora::Tensor sample(const cora::ContSet &S, cora::Rng &rng, int64_t N) {
    return S.rand_point(N, rng);
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
