// infimum - the lower bound matrix of an interval matrix, as CORA's intervalMatrix.infimum
//
// Syntax:   Tensor M = A.infimum();
// Inputs:   -
// Outputs:  M - the entrywise smallest matrix of the set
// See also: center, abs
//
// CORA: matrixSet/@intervalMatrix/infimum.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::infimum() const {
    detail::Entry entry;
    return int_.infimum();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
