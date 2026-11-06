// rad - the radius matrix of an interval matrix, as CORA's intervalMatrix.rad
//
// Syntax:   Tensor R = A.rad();
// Inputs:   -
// Outputs:  R - (sup - inf) / 2 entrywise: every matrix of the set is within R of the center
// See also: center, abs
//
// CORA: matrixSet/@intervalMatrix/rad.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::rad() const {
    detail::Entry entry;
    return int_.rad();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
