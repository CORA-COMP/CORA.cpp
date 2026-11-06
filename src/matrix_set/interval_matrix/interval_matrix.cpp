// interval_matrix - the constructors of an interval matrix, checked, as CORA's intervalMatrix
//
// Syntax:   IntervalMatrix A(C, D);   IntervalMatrix A(C);   IntervalMatrix A(I);
// Inputs:   C, D - center and delta, matrices (..., rows, cols) of one shape
//           I - an Interval of matrix bounds
// Outputs:  A - [C - |D|, C + |D|]; for C alone the point matrix C; for I the set of its bounds
// See also: Interval::interval_matrix
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [C - |D|, C + |D|] of two matrices of one shape.
Interval aux_box(const Tensor &C, const Tensor &D) {
    detail::Entry entry;
    CORA_CHECK(C.defined() && D.defined() && C.ndim() >= 2 && C.shape() == D.shape(),
               Error::wrong_input_in_constructor, [&] {
                   return "IntervalMatrix: the center and the delta must be matrices of one shape" +
                          (C.defined() && D.defined()
                               ? ", got " + shape_text(C.shape()) + " and " + shape_text(D.shape())
                               : std::string(", got an undefined tensor"));
               });
    const Tensor R = D.abs();
    return {C - R, C + R};
}

} // namespace


// ===========================================  MAIN  =========================================== //

IntervalMatrix::IntervalMatrix(const Tensor &center, const Tensor &delta)
    : int_(aux_box(center, delta)) {}

IntervalMatrix::IntervalMatrix(const Tensor &center) : int_(center, center) {}

IntervalMatrix::IntervalMatrix(Interval bounds) : int_(std::move(bounds)) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
