// linear_map - the linear map M * Z, as CORA's zonotope.mtimes
//
// Syntax:   Zres = Z.linear_map(M);   Zres = Z.linear_map(I);
// Inputs:   M - matrix (..., n, n)
//           I - interval matrix: an Interval with (..., n, n) bounds
// Outputs:  Zres - M * Z exactly; for an interval matrix, a zonotope enclosing [I] * Z
// See also: plus, linear_combination, Interval::linear_map
//
// References:
//    [1] M. Althoff, O. Stursberg, M. Buss. "Reachability analysis of linear systems with
//        uncertain parameters and inputs", 46th IEEE CDC, 2007, pp. 726-732, Thm. 4 (the
//        interval matrix times a zonotope).
//
// CORA: contSet/@zonotope/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    const int64_t inner = elem_rank_ == 1 ? dim() : c_.size(-2);
    CORA_CHECK(M.shape().size() >= 2 && M.shape().back() == inner, Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(inner) +
               " columns (the rows of the elements of Z), got the shape " + shape_text(M.shape());
    });
    CORA_CHECK(elem_rank_ <= 2, Error::not_supported,
               [] { return std::string("linear_map: elements of rank 1 or 2 only"); });
    if (elem_rank_ == 2) {
        // M X for every matrix X: the plain products, the batch dims of M broadcast
        return {M.matmul(c_), M.expand_dims(-3).matmul(G_), 2};
    }
    auto [c, G] = Tensor::zonotope_linear_map(M, c_, G_);
    return {std::move(c), std::move(G)};
}

Zonotope Zonotope::matmul_right(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.shape().size() >= 2 && M.size(-2) == c_.size(-1) && elem_rank_ <= 2,
               Error::dimension_mismatch, [&] {
                   return "matmul: the matrix needs " + std::to_string(c_.size(-1)) +
                          " rows (the last element dim of Z), got the shape " +
                          shape_text(M.shape());
               });
    if (elem_rank_ == 2) return {c_.matmul(M), G_.matmul(M.expand_dims(-3)), 2};
    // x @ M = M' x for every point x
    auto [c, G] = Tensor::zonotope_linear_map(M.matrix_transpose(), c_, G_);
    return {std::move(c), std::move(G)};
}

Zonotope operator*(double s, const Zonotope &Z) { return {Z.c_ * s, Z.G_ * s, Z.elem_rank_}; }

Zonotope Zonotope::linear_map(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1, Error::not_supported,
               [] { return std::string("linear_map: an interval matrix maps a vector zonotope"); });

    CORA_CHECK(
        I.elem_rank() == 2 && I.inf().shape().back() == dim(), Error::dimension_mismatch, [&] {
            return "linear_map: the interval matrix needs " + std::to_string(dim()) +
                   " columns (the dimension of Z), got the shape " + shape_text(I.inf().shape());
        });
    // The center matrix maps Z; the radius adds one axis-aligned generator per dimension.
    auto [c, G] = Tensor::zonotope_interval_map(I.inf(), I.sup(), c_, G_);
    return {std::move(c), std::move(G)};
}

Set Zonotope::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
