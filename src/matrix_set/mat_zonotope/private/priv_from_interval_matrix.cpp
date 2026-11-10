// priv_from_interval_matrix - an interval matrix as a matrix zonotope
//
// See also: priv_from_interval_matrix.h
//
// CORA.cpp only: a helper of the matrix zonotope constructor from an interval matrix.
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "matrix_set/mat_zonotope/private/priv_from_interval_matrix.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope priv_from_interval_matrix(const IntervalMatrix &A) {
    const Interval &I = A.interval();
    const int64_t n = A.rows(), m = A.cols();
    const Tensor radius = I.rad();
    // generator j holds the radius of the entry j (in column order) at that entry
    const Tensor diagonal = priv_stack_columns(radius).diag();
    const Tensor transposed = diagonal.reshape(priv_replace_tail(diagonal, 2, {n * m, m, n}));
    return MatZonotope(A.center(), transposed.matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
