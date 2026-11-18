// interval_matrix_testing - what the tests of the interval matrix operations share
//
// Syntax:   const Interval A = test::example();   test::for_each_backend_and_lean(body);
//           const Tensor M = test::member(A, rng);   auto B = test::random_matrix(rng, 3, 0.2);
// Provides: an interval matrix as CORA's intervalMatrix makes it (an Interval as it is, a point
//           matrix, or a center and a delta), a hand-computed 2 x 2 example, the lean backend
//           next to the others, a random matrix set and a random matrix inside a set
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// CORA's intervalMatrix(I) of an Interval I (the set itself), (C) a point matrix and (C, D) the
/// matrices C +- |D|.
inline cora::Interval interval_matrix(const cora::Interval &I) { return I; }
inline cora::Interval interval_matrix(const cora::Tensor &C) { return {C, C}; }
inline cora::Interval interval_matrix(const cora::Tensor &C, const cora::Tensor &D) {
    return cora::Interval::from_center_delta(C, D);
}

/// A = [[[1, 2], [-1, 1]], [[0, 3], [2, 2]]]: inf [[1, -1], [0, 2]], sup [[2, 1], [3, 2]].
inline cora::Interval example() {
    return {cora::Tensor::from_data({1, -1, 0, 2}, {2, 2}),
            cora::Tensor::from_data({2, 1, 3, 2}, {2, 2})};
}

/// The matrix of rows x cols numbers, row by row.
inline cora::Tensor matrix(const std::vector<double> &values, int64_t rows, int64_t cols) {
    return cora::Tensor::from_data(values, {rows, cols});
}

/// A random n x n matrix set: a center from N(0, 1/2) and half-widths from U[0, spread].
inline cora::Interval random_matrix(cora::Rng &rng, int64_t n, double spread) {
    std::vector<double> c(n * n), d(n * n);
    rng.normal(c.data(), c.size(), 0.5);
    rng.uniform(d.data(), d.size(), 0.0, spread);
    return interval_matrix(matrix(c, n, n), matrix(d, n, n));
}

/// A random matrix of the set: every entry uniform between its bounds.
inline cora::Tensor member(const cora::Interval &A, cora::Rng &rng) {
    const int64_t rows = A.inf().size(-2), cols = A.inf().size(-1);
    std::vector<double> u(rows * cols);
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    return A.inf() + (A.sup() - A.inf()).multiply(matrix(u, rows, cols));
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
