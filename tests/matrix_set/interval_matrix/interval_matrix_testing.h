// interval_matrix_testing - what the tests of the interval matrix operations share
//
// Syntax:   const IntervalMatrix A = test::example();   test::for_each_backend_and_lean(body);
//           const Tensor M = test::member(A, rng);   IntervalMatrix B = test::random_matrix(rng, 3, 0.2);
// Provides: a hand-computed 2 x 2 example, the lean backend next to the others, a random
//           matrix set around a random center and a random matrix inside a set

#pragma once

#include "cora/global/rng.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "testing.h"

namespace test {

/// A = [[[1, 2], [-1, 1]], [[0, 3], [2, 2]]]: inf [[1, -1], [0, 2]], sup [[2, 1], [3, 2]].
inline cora::IntervalMatrix example() {
    return cora::IntervalMatrix(
        cora::Interval(cora::Tensor::from_data({1, -1, 0, 2}, {2, 2}),
                       cora::Tensor::from_data({2, 1, 3, 2}, {2, 2})));
}

/// The matrix of rows x cols numbers, row by row.
inline cora::Tensor matrix(const std::vector<double> &values, int64_t rows, int64_t cols) {
    return cora::Tensor::from_data(values, {rows, cols});
}

/// A random n x n matrix set: a center from N(0, 1/2) and half-widths from U[0, spread].
inline cora::IntervalMatrix random_matrix(cora::Rng &rng, int64_t n, double spread) {
    std::vector<double> c(n * n), d(n * n);
    rng.normal(c.data(), c.size(), 0.5);
    rng.uniform(d.data(), d.size(), 0.0, spread);
    return cora::IntervalMatrix(matrix(c, n, n), matrix(d, n, n));
}

/// A random matrix of the set: every entry uniform between its bounds.
inline cora::Tensor member(const cora::IntervalMatrix &A, cora::Rng &rng) {
    std::vector<double> u(A.rows() * A.cols());
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    const cora::Tensor w = matrix(u, A.rows(), A.cols());
    return A.infimum() + (A.interval().sup() - A.infimum()).multiply(w);
}

/// Runs `body(name)` on every backend of for_each_backend and on the lean one.
template <class F>
void for_each_backend_and_lean(F body) {
    for_each_backend(body);
    const cora::BackendScope scope("lean");
    body("lean");
}

} // namespace test
