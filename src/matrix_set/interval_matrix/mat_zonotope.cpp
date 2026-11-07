// mat_zonotope - an interval matrix as a matrix zonotope, as CORA's intervalMatrix.matZonotope
//
// The center is the center matrix; generator k is the radius of the entry k (column-major) times
// the matrix with a one there: one generator per entry.
//
// Syntax:   MatZonotope M = A.mat_zonotope();
// Inputs:   -
// Outputs:  M - the same set of matrices, with rows * cols generators
// See also: MatZonotope::interval_matrix
//
// CORA: matrixSet/@intervalMatrix/matZonotope.m
// CORA authors: Matthias Althoff, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope IntervalMatrix::mat_zonotope() const {
    detail::Entry entry;
    // the radius is the larger distance from the center to a bound, as the center absorbs rounding
    const Tensor C = center();
    const Tensor r = (infimum() - C).abs().maximum((supremum() - C).abs());
    const int64_t n = rows(), m = cols(), h = n * m;
    // the one-hot generator matrices: generator j n + i has a one at (i, j)
    std::vector<double> ones(std::size_t(h * h), 0.0);
    for (int64_t j = 0; j < m; ++j)
        for (int64_t i = 0; i < n; ++i) ones[std::size_t(((j * n + i) * n + i) * m + j)] = 1.0;
    // the radius gets a generator axis of length one to broadcast over
    Shape shape;
    for (int64_t d = 0; d + 2 < r.ndim(); ++d) shape.push_back(r.shape()[std::size_t(d)]);
    shape.push_back(1);
    shape.push_back(n);
    shape.push_back(m);
    return {C, Tensor::like(r, ones, {h, n, m}).multiply(r.reshape(shape))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
