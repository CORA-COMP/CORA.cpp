// poly_zonotope_testing - what the tests of the polynomial zonotope share
//
// Syntax:   PolyZonotope P = test::random_poly(rng, n);   bool in = test::inside(P, points);
// Sets:     the center and the generators are standard normal; two factors with exponents 0..3
// See also: testing.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <cmath>
#include <map>
#include <vector>

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
    // the draws of CORA's columns, as rows
    const cora::Tensor c = normal(rng, n, 1), G = normal(rng, n, 4), GI = normal(rng, n, 2);
    return {c.squeeze(-1), G.matrix_transpose(), GI.matrix_transpose(), E, {1, 2}};
}

/// A fixed random polynomial zonotope of 2 dimensions (the seed 1).
inline cora::PolyZonotope sample_poly() {
    cora::Rng rng(1);
    return random_poly(rng, 2);
}

/// The point of P at the dependent factors alpha (value by identifier, 0 for those not given) and
/// the independent factors beta (one per independent generator, 0 for those not given).
inline std::vector<double> point(const cora::PolyZonotope &P,
                                 const std::map<int64_t, double> &alpha,
                                 const std::vector<double> &beta) {
    const int64_t n = P.dim(), h = P.num_dependent();
    std::vector<double> p = P.center().data();
    const std::vector<double> G = P.generators().data(), GI = P.independent_generators().data();
    for (int64_t j = 0; j < h; ++j) {
        double monomial = 1.0;
        for (std::size_t r = 0; r < P.ids().size(); ++r) {
            const auto at = alpha.find(P.ids()[r]);
            monomial *=
                std::pow(at == alpha.end() ? 0.0 : at->second, P.exponents()[std::size_t(j)][r]);
        }
        for (int64_t i = 0; i < n; ++i) p[std::size_t(i)] += G[std::size_t(j * n + i)] * monomial;
    }
    for (std::size_t k = 0; k < beta.size(); ++k)
        for (int64_t i = 0; i < n; ++i)
            p[std::size_t(i)] += GI[k * std::size_t(n) + std::size_t(i)] * beta[k];
    return p;
}

/// Whether every row of the points lies in the zonotope up to a small tolerance.
inline bool inside(const cora::Zonotope &Z, const cora::Tensor &points) {
    return Z.contains(points, "exact", 1e-8).all();
}

/// The points of a set: N random ones, one per row (N, n).
inline cora::Tensor sample(const cora::ContSet &S, cora::Rng &rng, int64_t N) {
    return S.rand_point(N, rng);
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
