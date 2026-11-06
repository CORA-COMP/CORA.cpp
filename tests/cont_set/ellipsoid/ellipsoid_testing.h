// ellipsoid_testing - what the tests of the ellipsoid share: random ellipsoids and points
//
// Syntax:   Ellipsoid E = test::random_ellipsoid(rng, n);   Tensor p = test::ellipsoid_points(E,
// rng, N); Sets:     full rank (Q = A A' + I/2) or of rank r (Q = A A', A of n x r), a normal
// center Points:   q + G u r, u of unit length and r uniform in [0, 1] (inside), r = 1 (on the
// boundary) See also: testing.h
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The matrix (rows, cols) of normal numbers from rng, on the current backend.
inline cora::Tensor normal_matrix(cora::Rng &rng, int64_t rows, int64_t cols, double scale = 1.0) {
    std::vector<double> data(std::size_t(rows * cols));
    rng.normal(data.data(), data.size(), scale);
    return cora::Tensor::from_data(data, {rows, cols});
}

/// A random ellipsoid of n dimensions: rank r (default n, then Q is positive definite).
inline cora::Ellipsoid random_ellipsoid(cora::Rng &rng, int64_t n, int64_t rank = -1) {
    const int64_t r = rank < 0 ? n : rank;
    const cora::Tensor A = normal_matrix(rng, n, r);
    cora::Tensor Q = A.matmul(A.matrix_transpose());
    if (rank < 0) Q = Q + A.eye_like(n) * 0.5;
    return {(Q + Q.matrix_transpose()) * 0.5, normal_matrix(rng, n, 1)};
}

/// Whether the 1 x 1 tensor is finite and within tol of x (test::close accepts an infinity).
inline bool finite_near(const cora::Tensor &t, double x, double tol = 1e-8) {
    return std::isfinite(t.item()) && std::abs(t.item() - x) <= tol;
}

/// The columns of P (n, N) whose mask entries are set, as one matrix (n, k).
inline cora::Tensor select_columns(const cora::Tensor &P, const std::vector<bool> &keep) {
    std::vector<int64_t> columns;
    for (std::size_t j = 0; j < keep.size(); ++j)
        if (keep[j]) columns.push_back(int64_t(j));
    return P.take(columns, 1);
}

/// N points (n, N) of the ellipsoid: q + G u r for random unit directions u and r in [0, 1], or
/// r = 1 (the boundary) if `boundary`. G keeps the generators of length above 1e-6 only: the root
/// of a zero singular value is rounding noise, which would lead off a degenerate ellipsoid.
inline cora::Tensor ellipsoid_points(const cora::Ellipsoid &E, cora::Rng &rng, int64_t N,
                                     bool boundary = false) {
    cora::Tensor G = E.generators();
    std::vector<bool> long_enough;
    for (const double length : G.square().sum(0, false).sqrt().data())
        long_enough.push_back(length > 1e-6);
    G = select_columns(G, long_enough);
    cora::Tensor U = normal_matrix(rng, G.shape()[1], N);
    U = U.divide(U.square().sum(0, true).sqrt());
    std::vector<double> radius(std::size_t(N), 1.0);
    if (!boundary) rng.uniform(radius.data(), radius.size(), 0.0, 1.0);
    const cora::Tensor r = cora::Tensor::from_data(radius, {1, N});
    return E.q() + G.matmul(U.multiply(r));
}

/// The ellipsoid norms of the columns of P around the center of E, on the host.
inline std::vector<double> norms_of(const cora::Ellipsoid &E, const cora::Tensor &P) {
    std::vector<double> norms;
    norms.reserve(std::size_t(P.shape()[1]));
    for (int64_t j = 0; j < P.shape()[1]; ++j)
        norms.push_back(E.ellipsoid_norm(P.slice(1, j, j + 1) - E.q()).item());
    return norms;
}

/// Whether every column of P is in E up to the slack in its norm.
inline bool all_inside(const cora::Ellipsoid &E, const cora::Tensor &P, double slack = 1e-9) {
    for (const double x : norms_of(E, P))
        if (!(x <= 1.0 + slack)) return false;
    return true;
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
