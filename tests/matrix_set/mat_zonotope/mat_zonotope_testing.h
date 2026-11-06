// mat_zonotope_testing - what the tests of the matrix zonotope operations share
//
// Syntax:   const MatZonotope A = test::example();   MatZonotope B = test::random_set(rng, 3, 2,
// 0.2);
//           const Tensor M = test::member(A, b);   std::vector<double> b = test::coefficients(rng,
//           h);
// Provides: a hand-computed 2 x 2 example with two generators, the lean backend next to the others,
//           a random square set, the matrix C + sum b_i G_i for given coefficients, a batch of two
//           sets and one set of a batch
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/rng.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The matrix of rows x cols numbers, row by row.
inline cora::Tensor matrix(const std::vector<double> &values, int64_t rows, int64_t cols) {
    return cora::Tensor::from_data(values, {rows, cols});
}

/// C = [[1, 2], [3, 4]] with the generators [[1, 0], [0, 1]] and [[0, 1], [1, 0]].
inline cora::MatZonotope example() {
    return cora::MatZonotope(matrix({1, 2, 3, 4}, 2, 2),
                             cora::Tensor::from_data({1, 0, 0, 1, 0, 1, 1, 0}, {2, 2, 2}));
}

/// A random n x n set of h generators: a center from N(0, 1/2) and entries from U[-spread, spread].
inline cora::MatZonotope random_set(cora::Rng &rng, int64_t n, int64_t h, double spread) {
    std::vector<double> c(n * n), g(h * n * n);
    rng.normal(c.data(), c.size(), 0.5);
    rng.uniform(g.data(), g.size(), -spread, spread);
    return cora::MatZonotope(matrix(c, n, n), cora::Tensor::from_data(g, {h, n, n}));
}

/// h coefficients of U[-1, 1].
inline std::vector<double> coefficients(cora::Rng &rng, int64_t h) {
    std::vector<double> b(h);
    rng.uniform(b.data(), b.size(), -1.0, 1.0);
    return b;
}

/// The matrix C + sum_i b_i G_i of the set.
inline cora::Tensor member(const cora::MatZonotope &A, const std::vector<double> &b) {
    const cora::Tensor weights = cora::Tensor::like(A.center(), b, {int64_t(b.size()), 1, 1});
    return A.center() + A.generators().multiply(weights).sum(-3);
}

/// The set of C and the generators G (h, rows, cols) given as numbers.
inline cora::MatZonotope set_of(const std::vector<double> &C, const std::vector<double> &G,
                                int64_t rows, int64_t cols) {
    return cora::MatZonotope(
        matrix(C, rows, cols),
        cora::Tensor::from_data(G, {int64_t(G.size()) / (rows * cols), rows, cols}));
}

/// exp(M) by the Taylor series to 40 terms, exact to rounding for the small matrices of the tests
/// (the lean backend has no expm).
inline cora::Tensor exponential(const cora::Tensor &M) {
    cora::Tensor sum = M.eye_like(M.size(-1)) + M.zeros_like(), term = sum;
    for (int k = 1; k <= 40; ++k) {
        term = term.matmul(M) * (1.0 / k);
        sum = sum + term;
    }
    return sum;
}

/// Whether the center and the generators of two sets agree up to tol.
inline bool same_set(const cora::MatZonotope &A, const cora::MatZonotope &B, double tol = 1e-9) {
    return close(A.center(), B.center(), tol) && close(A.generators(), B.generators(), tol);
}

/// The sets stacked as a batch of two (equal numbers of generators).
inline cora::MatZonotope batch_of(const cora::MatZonotope &A, const cora::MatZonotope &B) {
    return cora::MatZonotope(cora::Tensor::stack({A.center(), B.center()}, 0),
                             cora::Tensor::stack({A.generators(), B.generators()}, 0));
}

/// Member i of a batch.
inline cora::MatZonotope batch_member(const cora::MatZonotope &A, int64_t i) {
    return cora::MatZonotope(A.center().slice(0, i, i + 1).squeeze(0),
                             A.generators().slice(0, i, i + 1).squeeze(0));
}

/// Runs `body(name)` on every backend of for_each_backend and on the lean one.
template <class F>
void for_each_backend_and_lean(F body) {
    for_each_backend(body);
    const cora::BackendScope scope("lean");
    body("lean");
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
