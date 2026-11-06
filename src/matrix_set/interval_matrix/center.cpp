// center - the center matrix of an interval matrix, as CORA's intervalMatrix.center
//
// Syntax:   Tensor C = A.center();
// Inputs:   -
// Outputs:  C - (inf + sup) / 2, a matrix of the set
// See also: infimum, abs
//
// CORA: matrixSet/@intervalMatrix/center.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::center() const {
    detail::Entry entry;
    return int_.center();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
