// expm_absolute_bound - CORA's absolute bound of the mapping mimicking exp (expmAbsoluteBound)
//
// Syntax:   Tensor B = A.expm_absolute_bound(t);
// Inputs:   t >= 0 - the time;  A is square, with the spectral radii of |C| t / 2 and
//           (|C| + S) t / 2 below 1 (C the center, S the radius): then the eigenvalue 1 of
//           CORA's block matrix is the n-fold one of its last block
// Outputs:  B - (..., n, n): the last block row of lim G^k times [|C| t; S t; 0], G the block
//           matrix [|C| t / 2, 0, 0; S t / 2, (|C| + S) t / 2, 0; 0, I, I]. CORA forms the
//           spectral projector for the eigenvalue 1 from the eigenvectors of G and G'; here the
//           same limit comes from the block structure by two solves (no eigendecomposition)
// See also: expm_norm_infinity, abs
//
// CORA: matrixSet/@intervalMatrix/expmAbsoluteBound.m
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

Tensor IntervalMatrix::expm_absolute_bound(double t) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_absolute_bound: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(t >= 0, Error::wrong_value, [&] {
        return "expm_absolute_bound: the time must be nonnegative, got " + std::to_string(t);
    });
    const Tensor C = center(), S = rad(), I = C.eye_like(rows());
    const Tensor abs_C = C.abs();
    const double j = 2.0;
    const Tensor B1 = abs_C * t / j, B2 = (abs_C + S) * t / j, C21 = S * t / j;
    // x -> B1 x, y -> C21 x + B2 y, z -> y + z: the sums of the x and y sequences
    const Tensor Y = (I - B1).solve(abs_C * t);
    return (I - B2).solve(C21.matmul(Y) + S * t);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
