// linear_map - the linear map M * Z, as CORA's zonotope.mtimes
//
// Syntax:   Zres = Z.linear_map(M);   Zres = Z.linear_map(I);
// Inputs:   M - matrix (..., n, n)
//           I - interval matrix: an Interval with (..., n, n) bounds
// Outputs:  Zres - M * Z exactly; for an interval matrix, a zonotope enclosing [I] * Z
// See also: plus, linear_combination, Interval::linear_map
//
// CORA: contSet/@zonotope/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.shape().size() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of Z), got the shape " + shape_text(M.shape());
    });
    auto [c, G] = Tensor::zonotope_linear_map(M, c_, G_);
    return {std::move(c), std::move(G)};
}

Zonotope Zonotope::linear_map(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.inf().shape().size() >= 2 && I.inf().shape().back() == dim(),
               Error::dimension_mismatch, [&] {
                   return "linear_map: the interval matrix needs " + std::to_string(dim()) +
                          " columns (the dimension of Z), got the shape " +
                          shape_text(I.inf().shape());
               });
    // The center matrix maps Z; the radius adds one axis-aligned generator per dimension, of how
    // far it can move a point of Z: rad(I) (|c| + sum_j |G_j|).
    auto [c, G] = Tensor::zonotope_interval_map(I.inf(), I.sup(), c_, G_);
    return {std::move(c), std::move(G)};
}

Set Zonotope::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
