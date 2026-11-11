// linear_map - the image under a matrix or an interval matrix, as CORA's mtimes
//
// A matrix maps the center and the generators and keeps the factors. An interval matrix
// [M] = Mc + [-R, R] maps the center and the generators by Mc and adds the independent generators
// diag(R s) with s = |center| + radius of the box of the set, as CORA's intervalMatrix mtimes.
//
// Syntax:   S = pZ.linear_map(M);   S = pZ.linear_map(Mint);
// Inputs:   M - a matrix (k, n), or an interval matrix with (k, n) bounds
// Outputs:  S - the image (exact for a matrix, an enclosure for an interval matrix)
// See also: enclose
//
// CORA: contSet/@polyZonotope/mtimes.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
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
    return {M.matmul(c_), M.matmul(G_), M.matmul(GI_), E_, id_};
}

PolyZonotope PolyZonotope::linear_map(const Interval &M) const {
    detail::Entry entry;
    CORA_CHECK(M.inf().ndim() == 2 && M.inf().shape()[1] == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the interval matrix needs " + std::to_string(dim()) + " columns, got " +
               shape_text(M.inf().shape());
    });
    // the center and the radius of the matrix, and the largest |x| of the set per coordinate
    const Tensor Mc = (M.sup() + M.inf()) * 0.5, R = (M.sup() - M.inf()) * 0.5;
    const Interval box = interval();
    const Tensor center = (box.sup() + box.inf()) * 0.5, radius = (box.sup() - box.inf()) * 0.5;
    const Tensor S = center.abs() + radius;
    const Tensor GI = Tensor::concat({Mc.matmul(GI_), R.matmul(S).diag()}, -1);
    return {Mc.matmul(c_), Mc.matmul(G_), GI, E_, id_};
}

Set PolyZonotope::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
