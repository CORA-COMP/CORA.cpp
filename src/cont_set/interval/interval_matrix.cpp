// interval_matrix - the interval matrix of the center and radius, as CORA's interval.intervalMatrix
//
// Equal to I up to rounding (an Interval of matrix bounds is CORA's intervalMatrix); a box gives
// the interval matrix of one column.
//
// Syntax:   Interval M = I.interval_matrix();
// Inputs:   -
// Outputs:  M - [c - r, c + r] with the center c and the radius r of I
// See also: center, half_width
//
// CORA: contSet/@interval/intervalMatrix.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::interval_matrix() const {
    detail::Entry entry;
    // a box is the matrix of one column
    const Tensor c = elem_rank_ == 1 ? center().expand_dims(-1) : center(),
                 r = elem_rank_ == 1 ? half_width().expand_dims(-1) : half_width();
    return {c - r, c + r, 2, empty_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
