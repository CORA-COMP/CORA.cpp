// from_center_delta - the interval matrix of a center and a delta, as CORA's intervalMatrix(C, D)
//
// CORA's constructor intervalMatrix(C, D) takes the center and the delta, not the bounds; an
// Interval of matrix bounds is the interval matrix itself (CORA's intervalMatrix of an interval).
//
// Syntax:   Interval A = Interval::from_center_delta(C, D);
// Inputs:   C, D - center and delta, matrices (..., rows, cols) of one shape
// Outputs:  A - the interval matrix [C - |D|, C + |D|]
// See also: interval_matrix
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::from_center_delta(const Tensor &center, const Tensor &delta) {
    detail::Entry entry;
    CORA_CHECK(center.defined() && delta.defined() && center.ndim() >= 2 &&
                   center.shape() == delta.shape(),
               Error::wrong_input_in_constructor, [&] {
                   return "from_center_delta: the center and the delta must be matrices of one "
                          "shape" +
                          (center.defined() && delta.defined()
                               ? ", got " + shape_text(center.shape()) + " and " +
                                     shape_text(delta.shape())
                               : std::string(", got an undefined tensor"));
               });
    // the delta is a radius up to its sign
    const Tensor R = delta.abs();
    return matrix(center - R, center + R);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
