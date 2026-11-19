// kron - the Kronecker product of interval matrices, as CORA's interval.kron
//
// Block (i, j) is [a_ij] * [B]; a numeric factor has the one corner.
//
// Syntax:   Interval P = I.kron(I2);   Interval P = I.kron(B);   Interval P = Interval::kron(A, I);
// Inputs:   I2 - interval matrix
//           A, B - numeric matrices (..., m, n)
// Outputs:  P - the entrywise hull of the Kronecker products at the corners
// See also: matrix_power
//
// CORA: contSet/@interval/kron.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::kron(const Interval &I2) const {
    detail::Entry entry;
    const Tensor a = inf_matrix(), b = sup_matrix(), a2 = I2.inf_matrix(), b2 = I2.sup_matrix();
    const Tensor c1 = a.kron(a2), c2 = a.kron(b2), c3 = b.kron(a2), c4 = b.kron(b2);
    const Interval K = Interval::matrix(c1.minimum(c2).minimum(c3.minimum(c4)),
                                        c1.maximum(c2).maximum(c3.maximum(c4)));
    // two boxes (columns) give a box
    return elem_rank_ == 1 && I2.elem_rank_ == 1 ? from_matrix(K.inf_, K.sup_) : K;
}

Interval Interval::kron(const Tensor &B) const {
    detail::Entry entry;
    CORA_CHECK(B.ndim() >= 2, Error::dimension_mismatch,
               [] { return std::string("kron: the numeric factor must be a matrix"); });
    const Tensor c1 = inf_matrix().kron(B), c2 = sup_matrix().kron(B);
    return Interval::matrix(c1.minimum(c2), c1.maximum(c2));
}

Interval Interval::kron(const Tensor &A, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(A.ndim() >= 2, Error::dimension_mismatch,
               [] { return std::string("kron: the numeric factor must be a matrix"); });
    const Tensor c1 = A.kron(I.inf_matrix()), c2 = A.kron(I.sup_matrix());
    return Interval::matrix(c1.minimum(c2), c1.maximum(c2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
