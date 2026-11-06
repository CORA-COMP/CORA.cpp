// center - the center matrix of a matrix zonotope, as CORA's matZonotope.center
//
// Syntax:   Tensor C = A.center();
// Inputs:   -
// Outputs:  C - the center matrix (..., rows, cols)
// See also: generators, interval_matrix
//
// CORA: matrixSet/@matZonotope/center.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor MatZonotope::center() const {
    detail::Entry entry;
    return C_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
