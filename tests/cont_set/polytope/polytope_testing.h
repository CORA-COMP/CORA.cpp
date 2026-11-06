// polytope_testing - what the tests of the polytope share: random polytopes and points
//
// Syntax:   Polytope P = test::random_polytope(rng, n, m);   Tensor X = test::sphere_points(rng, n,
// k);
//           Tensor pts = test::sample_points(P, rng, N);   Tensor V = test::vertices_of(P);
// Sets:     random_polytope: m random rows with positive offsets (the origin inside) and the box
//           |x| <= 3, so bounded and nonempty; sphere_points: k points on a sphere, all extreme;
//           sample_points: points of a halfspace polytope by rejection in its box
// See also: testing.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/polytope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// A bounded polytope of n dimensions in halfspace form with m random rows and the box |x| <= 3.
inline cora::Polytope random_polytope(cora::Rng &rng, int n, int m) {
    std::vector<double> A(static_cast<std::size_t>(m * n)), b(static_cast<std::size_t>(m));
    rng.normal(A.data(), A.size(), 1.0);
    rng.uniform(b.data(), b.size(), 0.5, 2.0);
    for (int i = 0; i < n; ++i) {
        for (const double sign : {1.0, -1.0}) {
            for (int j = 0; j < n; ++j) A.push_back(i == j ? sign : 0.0);
            b.push_back(3.0);
        }
    }
    const int64_t rows = int64_t(b.size());
    return {cora::Tensor::from_data(A, {rows, n}), cora::Tensor::from_data(b, {rows, 1})};
}

/// The vertices of a bounded polytope (n, p), from either form.
inline cora::Tensor vertices_of(const cora::Polytope &P) {
    const cora::Tensor &like = P.is_h_rep() ? P.A() : P.V();
    return cora::priv_points_tensor(like, P.dim(), cora::priv_polytope_vertices(P));
}

/// A bounded polytope as the hull of its vertices.
inline cora::Polytope as_vertices(const cora::Polytope &P) {
    return cora::Polytope(vertices_of(P));
}

/// The unit square [0, 1]^2 by its halfspaces, and by its vertices.
inline cora::Polytope square_h() {
    return {cora::Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
            cora::Tensor::from_data({1.0, 1.0, 0.0, 0.0}, {4, 1})};
}

inline cora::Polytope square_v() {
    return cora::Polytope(cora::Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
}

/// The triangle x >= 0, y >= 0, x + y <= 2 by its halfspaces.
inline cora::Polytope triangle_h() {
    return {cora::Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}),
            cora::Tensor::from_data({0.0, 0.0, 2.0}, {3, 1})};
}

/// k points (n, k) on the sphere of the radius about the point (shift, ..., shift): all vertices.
inline cora::Tensor sphere_points(cora::Rng &rng, int n, int k, double radius = 1.0,
                                  double shift = 0.0) {
    std::vector<double> points(std::size_t(n) * k);
    rng.normal(points.data(), points.size(), 1.0);
    for (int c = 0; c < k; ++c) {
        double norm = 0.0;
        for (int i = 0; i < n; ++i)
            norm += points[std::size_t(i) * k + c] * points[std::size_t(i) * k + c];
        norm = std::sqrt(norm);
        for (int i = 0; i < n; ++i)
            points[std::size_t(i) * k + c] = shift + radius * points[std::size_t(i) * k + c] / norm;
    }
    return cora::Tensor::from_data(points, {n, k});
}

/// N points (n, N) of a bounded halfspace polytope with inequalities only: uniform in its box,
/// kept if they satisfy the rows.
inline cora::Tensor sample_points(const cora::Polytope &P, cora::Rng &rng, int N) {
    const cora::Interval box = P.interval();
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    const std::vector<double> A = P.A().data(), b = P.b().data();
    const int64_t n = P.dim(), m = P.A().shape()[0];
    std::vector<std::vector<double>> kept;
    while (int(kept.size()) < N) {
        std::vector<double> x(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            rng.uniform(&x[std::size_t(i)], 1, lo[std::size_t(i)], hi[std::size_t(i)]);
        bool inside = true;
        for (int64_t r = 0; r < m && inside; ++r) {
            double value = 0.0;
            for (int64_t i = 0; i < n; ++i) value += A[std::size_t(r * n + i)] * x[std::size_t(i)];
            inside = value <= b[std::size_t(r)];
        }
        if (inside) kept.push_back(x);
    }
    std::vector<double> flat(std::size_t(n) * N);
    // the points as a matrix
    for (int k = 0; k < N; ++k)
        for (int64_t i = 0; i < n; ++i)
            flat[std::size_t(i) * N + k] = kept[std::size_t(k)][std::size_t(i)];
    return cora::Tensor::from_data(flat, {n, int64_t(N)});
}

/// Whether the columns of P (n, N) all satisfy the halfspaces up to slack.
inline bool inside_halfspaces(const cora::Tensor &points, const cora::Polytope::Constraints &C,
                              double slack = 1e-8) {
    const cora::Tensor ineq = C.A.matmul(points) - C.b;
    if (ineq.numel() > 0 && ineq.max().item() > slack) return false;
    const cora::Tensor eq = (C.Ae.matmul(points) - C.be).abs();
    return eq.numel() == 0 || eq.max().item() <= slack;
}

/// The columns of a matrix (n, p) as sorted rows, for comparisons of vertex sets that do not
/// depend on the order.
inline std::vector<std::vector<double>> columns_sorted(const cora::Tensor &V) {
    const int64_t n = V.shape()[0], p = V.shape()[1];
    const std::vector<double> flat = V.data();
    std::vector<std::vector<double>> columns(static_cast<std::size_t>(p),
                                             std::vector<double>(static_cast<std::size_t>(n)));
    for (int64_t k = 0; k < p; ++k)
        for (int64_t i = 0; i < n; ++i)
            columns[std::size_t(k)][std::size_t(i)] = flat[std::size_t(i * p + k)];
    std::sort(columns.begin(), columns.end());
    return columns;
}

/// Whether two sets of columns agree up to tol, whatever the order.
inline bool same_columns(const cora::Tensor &A, const cora::Tensor &B, double tol = 1e-9) {
    if (A.shape() != B.shape()) return false;
    const auto a = columns_sorted(A), b = columns_sorted(B);
    for (std::size_t k = 0; k < a.size(); ++k)
        for (std::size_t i = 0; i < a[k].size(); ++i)
            if (!close(a[k][i], b[k][i], tol)) return false;
    return true;
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
