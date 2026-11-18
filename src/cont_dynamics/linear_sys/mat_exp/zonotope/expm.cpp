// expm - the matrix exponential of a matrix zonotope, as CORA's matZonotope.expm
//
// CORA's expm calls expmInd.
//
// Syntax:   Zonotope E = mat_exp::expm(A);   Zonotope E = mat_exp::expm(A, max_order);
// Inputs:   max_order - the order of the series (default 10);  A is square
// Outputs:  E - an enclosure of exp(M) for every matrix M of A: expm_independent
// See also: expm_independent, expm_mixed
//
// CORA: matrixSet/@matZonotope/expm.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope mat_exp::expm(const Zonotope &A, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "expm: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm: the order must be nonnegative, got " + std::to_string(max_order);
    });
    return expm_independent(A, max_order);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
