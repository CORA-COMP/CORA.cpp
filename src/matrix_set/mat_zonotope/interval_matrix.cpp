// interval_matrix - the box around a matrix zonotope as an interval matrix, as CORA's
//
// The smallest box around the set.
//
// Syntax:   IntervalMatrix B = A.interval_matrix();
// Inputs:   -
// Outputs:  B - the interval matrix [C - D, C + D] with D the sum of the absolute generators
// See also: norm, zonotope
//
// CORA: matrixSet/@matZonotope/intervalMatrix.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix MatZonotope::interval_matrix() const {
    detail::Entry entry;
    return IntervalMatrix(C_, G_.abs().sum(-3));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
