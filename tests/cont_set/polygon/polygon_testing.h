// polygon_testing - what the tests of the polygon share: fixed and random polygons, points
//
// Syntax:   Polygon L = test::l_shape();   Polygon P = test::random_polygon(rng, 3);
//           Tensor X = test::random_points(rng, N, -3, 3);
// Sets:     the unit square, the L of three unit squares (area 3), random unions of convex pieces
//           (the hulls of random points about random centers)
// See also: testing.h
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/polygon/polygon.h"
#include "cora/cont_set/registry.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The polygon of the boundary rows.
inline cora::Polygon boundary(const std::vector<std::vector<double>> &rows) {
    std::vector<double> data;
    for (const std::vector<double> &r : rows) data.insert(data.end(), r.begin(), r.end());
    return cora::Polygon(cora::Tensor::from_data(data, {int64_t(rows.size()), 2}));
}

/// The unit square [0, 1]^2, one piece.
inline cora::Polygon unit_square() { return boundary({{0, 0}, {1, 0}, {1, 1}, {0, 1}}); }

/// The L [0, 2] x [0, 1] with [0, 1] x [1, 2] (area 3), not convex.
inline cora::Polygon l_shape() {
    return boundary({{0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}});
}

/// A convex piece: the polytope of k random points about c.
inline cora::Polytope random_piece(cora::Rng &rng, double cx, double cy, int k = 6) {
    std::vector<double> x(2 * std::size_t(k));
    rng.normal(x.data(), x.size(), 1.0);
    for (int i = 0; i < k; ++i) x[2 * i] += cx, x[2 * i + 1] += cy;
    return cora::Polytope(cora::Tensor::from_data(x, {k, 2}));
}

/// A union of m random convex pieces about centers in [-2, 2]^2.
inline cora::Polygon random_polygon(cora::Rng &rng, int m, int k = 6) {
    std::vector<double> c(2 * std::size_t(m));
    rng.uniform(c.data(), c.size(), -2, 2);
    std::vector<cora::Polytope> pieces;
    pieces.reserve(std::size_t(m));
    for (int i = 0; i < m; ++i) pieces.push_back(random_piece(rng, c[2 * i], c[2 * i + 1], k));
    return cora::Polygon(pieces);
}

/// N points (N, 2) uniform in [lo, hi]^2.
inline cora::Tensor random_points(cora::Rng &rng, int N, double lo, double hi) {
    std::vector<double> x(2 * std::size_t(N));
    rng.uniform(x.data(), x.size(), lo, hi);
    return cora::Tensor::from_data(x, {N, 2});
}

/// Point i of the rows (N, 2).
inline cora::Tensor row(const cora::Tensor &X, int i) {
    const std::vector<double> x = X.data();
    return vec({x[2 * std::size_t(i)], x[2 * std::size_t(i) + 1]});
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
