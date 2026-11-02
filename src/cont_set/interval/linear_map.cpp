// linear_map - the linear map M * I of a box, as CORA's interval.mtimes
//
// Syntax:   Ires = I.linear_map(M);
// Inputs:   M - matrix (..., n, n)
// Outputs:  Ires - the interval hull of M * I: center M*c, radius |M|*r
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
    const Tensor c = M.matmul(center());
    const Tensor r = M.abs().matmul(rad());
    return {c - r, c + r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
