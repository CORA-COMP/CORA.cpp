// linear_map - the linear map M * I of a box, as CORA's interval.mtimes
//
// Each bound sums the smaller (larger) of m_ik inf_k and m_ik sup_k.
//
// Syntax:   Ires = I.linear_map(M);
// Inputs:   M - matrix (..., n, n)
// Outputs:  Ires - the interval hull of M * I, as CORA
// See also: Zonotope::linear_map
// References: none (CORA cites only a thesis)
//
// CORA: contSet/@interval/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.shape().size() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of I), got the shape " + shape_text(M.shape());
    });
    // a box is the column (n, 1) of the product
    auto [lo, hi] = Tensor::matmul_interval(M, inf_matrix(), sup_matrix());
    return from_matrix(std::move(lo), std::move(hi));
}

Set Interval::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
