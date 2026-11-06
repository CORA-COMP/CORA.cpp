// supremum - the upper bound matrix of an interval matrix, as CORA's intervalMatrix.supremum
//
// Syntax:   Tensor M = A.supremum();
// Inputs:   -
// Outputs:  M - the entrywise largest matrix of the set
// See also: infimum, center
//
// CORA: matrixSet/@intervalMatrix/supremum.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::supremum() const {
    detail::Entry entry;
    return int_.supremum();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
