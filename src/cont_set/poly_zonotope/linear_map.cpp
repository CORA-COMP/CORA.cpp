// linear_map - the image under a matrix or an interval matrix, as CORA's mtimes
//
// A matrix maps the center and the generators and keeps the factors. An interval matrix
// [M] = Mc + [-R, R] maps the center and the generators by Mc and adds the independent generators
// diag(R s) with s = |center| + radius of the box of the set, as CORA's intervalMatrix mtimes
// (Mc is CORA's 0.5 (sup + inf), not Interval::center: it keeps an infinite entry).
//
// Syntax:   S = pZ.linear_map(M);   S = pZ.linear_map(Mint);
// Inputs:   M - a matrix (k, n), or an interval matrix with (k, n) bounds
// Outputs:  S - the image (exact for a matrix, an enclosure for an interval matrix)
// See also: enclose
//
// CORA: contSet/@polyZonotope/mtimes.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() == 2 && M.shape()[1] == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns, got the shape " + shape_text(M.shape());
    });
    // on the rows: c M' and G M', GI M'
    const auto [c, G] = Tensor::zonotope_linear_map(M, c_, G_);
    return {c, G, GI_.matmul(M.matrix_transpose()), E_, id_, detail::Normalized{}};
}

PolyZonotope PolyZonotope::linear_map(const Interval &M) const {
    detail::Entry entry;
    CORA_CHECK(M.elem_rank() == 2 && M.inf().ndim() == 2 && M.inf().shape()[1] == dim(),
               Error::dimension_mismatch, [&] {
                   return "linear_map: the interval matrix needs " + std::to_string(dim()) +
                          " columns, got " + shape_text(M.inf().shape());
               });
    // the center and radius of the matrix; the largest |x| of the set, |center| + rad of its box
    const Tensor Mc = (M.sup() + M.inf()) * 0.5, R = M.half_width();
    const Interval box = interval();
    const Tensor S = (box.center().abs() + box.half_width()).expand_dims(-1);
    const Tensor GI = Tensor::concat({GI_.matmul(Mc.matrix_transpose()), R.matmul(S).diag()}, -2);
    const auto [c, G] = Tensor::zonotope_linear_map(Mc, c_, G_);
    return {c, G, GI, E_, id_, detail::Normalized{}};
}

Set PolyZonotope::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
