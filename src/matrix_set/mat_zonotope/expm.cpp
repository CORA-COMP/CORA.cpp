// expm - the matrix exponential of a matrix zonotope, as CORA's matZonotope.expm
//
// Syntax:   MatZonotope E = A.expm();   MatZonotope E = A.expm(max_order);
// Inputs:   max_order - the order of the series (default 10);  A is square
// Outputs:  E - an enclosure of exp(M) for every matrix M of A: expm_independent, as CORA's expm
//           calls expmInd
// See also: expm_independent, expm_mixed
//
// CORA: matrixSet/@matZonotope/expm.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::expm(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm: the matrix zonotope must be square, got the shape " + shape_text(C_.shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm: the order must be nonnegative, got " + std::to_string(max_order);
    });
    return expm_independent(max_order);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
