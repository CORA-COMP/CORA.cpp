// linear_map - the linear map M * I of a box, as CORA's interval.mtimes
//
// Syntax:   Ires = I.linear_map(M);
// Inputs:   M - matrix (..., n, n)
// Outputs:  Ires - the interval hull of M * I: each bound sums the smaller (larger) of
//           m_ik inf_k and m_ik sup_k, as CORA
// See also: Zonotope::linear_map
//
// CORA: contSet/@interval/mtimes.m

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
    auto [lo, hi] = Tensor::matmul_interval(M, inf_, sup_);
    return {std::move(lo), std::move(hi)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
