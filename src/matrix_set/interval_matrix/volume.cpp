// volume - the volume of an interval matrix, as CORA's intervalMatrix.volume
//
// That is Interval::volume; 0 if some width is 0 or the matrix has no entries.
//
// Syntax:   Tensor v = A.volume();
// Inputs:   -
// Outputs:  v - (..., 1, 1): the product of the entry widths (CORA's volume of the stacked box)
// See also: rad, Interval::volume
//
// CORA: matrixSet/@intervalMatrix/volume.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "matrix_set/interval_matrix/private/priv_vec.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::volume() const {
    detail::Entry entry;
    const Tensor v = priv_vec(int_).volume();
    return rows() * cols() == 0 ? v.zeros_like() : v;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
