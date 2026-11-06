// plus - the sum of an interval matrix with a set, a matrix or a number, as CORA's plus
//
// Syntax:   IntervalMatrix C = A.plus(B);   C = A.plus(M);   C = A.plus(s);   C = A + B;
// Inputs:   B - interval matrix
//           M - matrix of numbers
//           s - number;  of the shape of A
// Outputs:  C - the Minkowski sum: the bounds add (a matrix or a number is a point). CORA's plus
//           is commutative, so M + A and s + A are the same sets
// See also: matmul, Interval::plus
//
// CORA: matrixSet/@intervalMatrix/plus.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix IntervalMatrix::plus(const IntervalMatrix &B) const {
    detail::Entry entry;
    CORA_CHECK(rows() == B.rows() && cols() == B.cols(), Error::dimension_mismatch, [&] {
        return "plus: the matrices differ in shape, " + shape_text(int_.inf().shape()) + " and " +
               shape_text(B.int_.inf().shape());
    });
    return IntervalMatrix(int_.plus(B.int_));
}

IntervalMatrix IntervalMatrix::plus(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-2) == rows() && M.size(-1) == cols(),
               Error::dimension_mismatch, [&] {
                   return "plus: the matrix must have the shape " + shape_text(int_.inf().shape()) +
                          ", got " + shape_text(M.shape());
               });
    return IntervalMatrix(int_ + M);
}

IntervalMatrix IntervalMatrix::plus(double s) const {
    detail::Entry entry;
    return IntervalMatrix(Interval(int_.inf() + s, int_.sup() + s));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
