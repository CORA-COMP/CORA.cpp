// plus - the sum of a matrix zonotope with a matrix zonotope or a matrix, as CORA's plus
//
// Syntax:   MatZonotope S = A.plus(B);   MatZonotope S = A.plus(M);   S = A + B;
// Inputs:   B - matrix zonotope of the same rows and columns
//           M - matrix of numbers (rows, cols),
//           or 1 x 1, which is added to every entry
// Outputs:  S - for a matrix zonotope the Minkowski sum: the centers add and the generators are
//           joined (A's first); for a matrix the set shifted by it
// See also: matmul
//
// CORA: matrixSet/@matZonotope/plus.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::plus(const MatZonotope &B) const {
    detail::Entry entry;
    CORA_CHECK(rows() == B.rows() && cols() == B.cols(), Error::dimension_mismatch, [&] {
        return "plus: the matrix zonotopes must have the same rows and columns, got " +
               shape_text(C_.shape()) + " and " + shape_text(B.C_.shape());
    });
    return MatZonotope(C_ + B.C_, priv_join(G_, B.G_));
}

MatZonotope MatZonotope::plus(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && ((M.size(-2) == rows() && M.size(-1) == cols()) ||
                                 (M.size(-2) == 1 && M.size(-1) == 1 && rows() * cols() >= 1)),
               Error::dimension_mismatch, [&] {
                   return "plus: the matrix must be (" + std::to_string(rows()) + ", " +
                          std::to_string(cols()) + ") or 1 x 1, got the shape " +
                          shape_text(M.shape());
               });
    return MatZonotope(C_ + M, G_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
