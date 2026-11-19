// from_interval_matrix - an interval matrix as a matrix zonotope, as CORA's matZonotope
//
// The center is the center matrix; generator k is the radius of the entry k (column-major) times
// the matrix with a one there: one generator per entry.
//
// Syntax:   Zonotope M = Zonotope::from_interval_matrix(A);
// Inputs:   A - interval matrix: an Interval of matrix bounds (..., rows, cols)
// Outputs:  M - the same set of matrices, with rows * cols generators
// See also: Zonotope::matrix, interval
//
// CORA: matrixSet/@intervalMatrix/matZonotope.m
// CORA authors: Matthias Althoff, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::from_interval_matrix(const Interval &A) {
    detail::Entry entry;
    // an empty interval has no members, a matrix zonotope always has its center
    CORA_CHECK(!A.is_empty(), Error::not_supported, [&] {
        return "from_interval_matrix: the empty interval of the shape " +
               shape_text(A.inf().shape()) + " has no matrix zonotope";
    });
    // the radius is the larger distance from the center to a bound, as the center absorbs rounding
    const Tensor C = A.center();
    const Tensor r = (A.inf() - C).abs().maximum((A.sup() - C).abs());
    const int64_t n = A.inf().size(-2), m = A.inf().size(-1), h = n * m;
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
    return matrix(C, Tensor::like(r, ones, {h, n, m}).multiply(r.reshape(shape)));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
