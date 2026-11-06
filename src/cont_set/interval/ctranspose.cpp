// ctranspose - the transpose of an interval matrix, as CORA's interval.ctranspose
//
// The bounds are real, so the conjugate is the identity; nothing rounds.
//
// Syntax:   Interval R = I.ctranspose();
// Inputs:   -
// Outputs:  R - the bounds transposed
// See also: transpose
//
// CORA: contSet/@interval/ctranspose.m
// CORA authors: Dmitry Grebenyuk
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::ctranspose() const {
    detail::Entry entry;
    return transpose();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
