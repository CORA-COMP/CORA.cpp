// kron - the Kronecker product of interval matrices, as CORA's interval.kron
//
// Syntax:   Interval P = I.kron(I2);   Interval P = I.kron(B);   Interval P = Interval::kron(A, I);
// Inputs:   I2 - interval matrix
//           A, B - numeric matrices (..., m, n)
// Outputs:  P - the entrywise hull of the Kronecker products at the corners: block (i, j) is
//           [a_ij] * [B]; a numeric factor has the one corner
// See also: matrix_power
//
// CORA: contSet/@interval/kron.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::kron(const Interval &I2) const {
    detail::Entry entry;
    const Tensor c1 = inf_.kron(I2.inf()), c2 = inf_.kron(I2.sup());
    const Tensor c3 = sup_.kron(I2.inf()), c4 = sup_.kron(I2.sup());
    return {c1.minimum(c2).minimum(c3.minimum(c4)), c1.maximum(c2).maximum(c3.maximum(c4))};
}

Interval Interval::kron(const Tensor &B) const {
    detail::Entry entry;
    CORA_CHECK(B.ndim() >= 2, Error::dimension_mismatch,
               [] { return std::string("kron: the numeric factor must be a matrix"); });
    const Tensor c1 = inf_.kron(B), c2 = sup_.kron(B);
    return {c1.minimum(c2), c1.maximum(c2)};
}

Interval Interval::kron(const Tensor &A, const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(A.ndim() >= 2, Error::dimension_mismatch,
               [] { return std::string("kron: the numeric factor must be a matrix"); });
    const Tensor c1 = A.kron(I.inf()), c2 = A.kron(I.sup());
    return {c1.minimum(c2), c1.maximum(c2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
