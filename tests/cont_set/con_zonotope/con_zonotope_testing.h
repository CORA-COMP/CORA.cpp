// con_zonotope_testing - what the tests of the constrained zonotope share: random sets and points
//
// Syntax:   ConZonotope cz = test::random_con_zonotope(rng, n, m, p);
//           Tensor points = test::sample_points(cz, rng, N);
// Sets:     the constraints are [I B] x = b, the first p factors determined by the others, so that
//           points of the set are drawn by rejection; b comes from a factor vector inside the box,
//           so the set is nonempty
// See also: testing.h
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// A random constrained zonotope of n dimensions, m generators and p <= m constraints [I B].
inline cora::ConZonotope random_con_zonotope(cora::Rng &rng, int n, int m, int p) {
    std::vector<double> c(n), G(std::size_t(n) * m), B(std::size_t(p) * (m - p)), xi(m);
    rng.normal(c.data(), c.size(), 1.0);
    rng.normal(G.data(), G.size(), 1.0);
    rng.normal(B.data(), B.size(), 0.3);
    rng.uniform(xi.data(), xi.size(), -0.5, 0.5);
    std::vector<double> A(std::size_t(p) * m, 0.0), b(p, 0.0);
    for (int i = 0; i < p; ++i) {
        A[i * m + i] = 1.0;
        b[i] = xi[i];
        for (int j = 0; j < m - p; ++j) {
            A[i * m + p + j] = B[i * (m - p) + j];
            b[i] += B[i * (m - p) + j] * xi[p + j];
        }
    }
    const auto tensor = [](const std::vector<double> &v, int64_t r, int64_t k) {
        return cora::Tensor::from_data(v, {r, k});
    };
    return {tensor(c, n, 1), tensor(G, n, m), tensor(A, p, m), tensor(b, p, 1)};
}

/// The hexagon {(a, b) | |a| <= 1, |b| <= 1, |a + b| <= 1} as x = G xi with xi_1 + xi_2 + xi_3 = 0
/// (a = -xi_2, b = -xi_1): the vertices are (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1).
inline cora::ConZonotope hexagon() {
    return {cora::Tensor::from_data({0.0, 0.0}, {2, 1}),
            cora::Tensor::from_data({1.0, 0.0, 1.0, 0.0, 1.0, 1.0}, {2, 3}),
            cora::Tensor::from_data({1.0, 1.0, 1.0}, {1, 3}),
            cora::Tensor::from_data({0.0}, {1, 1})};
}

/// N points of the set (n, N): the free factors are uniform in the box, the others follow from
/// the constraints [I B] x = b (the set must be of the form of random_con_zonotope), by rejection.
inline cora::Tensor sample_points(const cora::ConZonotope &cz, cora::Rng &rng, int N) {
    const int64_t n = cz.dim(), m = cz.generators().shape().back(), p = cz.A().shape()[0];
    const std::vector<double> c = cz.c().data(), G = cz.generators().data(), A = cz.A().data(),
                              b = cz.b().data();
    std::vector<double> points;
    std::vector<std::vector<double>> columns;
    while (int(columns.size()) < N) {
        std::vector<double> xi(m);
        rng.uniform(xi.data(), xi.size(), -1.0, 1.0);
        bool inside = true;
        for (int64_t i = 0; i < p; ++i) {
            xi[i] = b[i];
            for (int64_t j = p; j < m; ++j) xi[i] -= A[i * m + j] * xi[j];
            inside = inside && std::abs(xi[i]) <= 1.0;
        }
        if (!inside) continue;
        std::vector<double> x(c);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < m; ++j) x[i] += G[i * m + j] * xi[j];
        // the points of the set
        columns.push_back(x);
    }
    points.resize(std::size_t(n) * N);
    for (int k = 0; k < N; ++k)
        for (int64_t i = 0; i < n; ++i) points[i * N + k] = columns[k][i];
    return cora::Tensor::from_data(points, {n, int64_t(N)});
}

/// Whether the column points of P are inside the box [inf, sup] up to slack.
inline bool inside_box(const cora::Tensor &P, const cora::Interval &I, double slack = 1e-9) {
    const std::vector<double> x = P.data(), lo = I.inf().data(), hi = I.sup().data();
    const int64_t n = P.shape()[0], N = P.shape()[1];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < N; ++k)
            if (x[i * N + k] < lo[i] - slack || x[i * N + k] > hi[i] + slack) return false;
    return true;
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
