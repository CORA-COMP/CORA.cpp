// transpose - the transpose of an interval matrix, as CORA's interval.transpose
//
// Syntax:   Interval R = I.transpose();
// Inputs:   -
// Outputs:  R - the bounds transposed; nothing rounds
// See also: ctranspose, Tensor::matrix_transpose
//
// CORA: contSet/@interval/transpose.m
// CORA authors: Dmitry Grebenyuk
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::transpose() const {
    detail::Entry entry;
    return {inf_.matrix_transpose(), sup_.matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
