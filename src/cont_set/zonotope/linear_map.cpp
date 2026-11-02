// linear_map - the linear map M * Z, as CORA's zonotope.mtimes
//
// Syntax:   Zres = Z.linear_map(M);   Zres = Z.linear_map(I);
// Inputs:   M - matrix (..., n, n);  I - interval matrix: an Interval with (..., n, n) bounds
// Outputs:  Zres - M * Z exactly; for an interval matrix, a zonotope enclosing [I] * Z
// See also: plus, linear_combination, Interval::linear_map
//
// CORA: contSet/@zonotope/mtimes.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// How far the radius of I can move a point of Z along each axis: rad(I) * (|c| + sum_j |G_j|).
Tensor aux_widening(const Zonotope &Z, const Interval &I) {
    const Tensor reach = Z.center().abs() + Z.generators().abs().sum_last();
    return I.rad().matmul(reach);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.shape().size() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of Z), got the shape " + shape_text(M.shape());
    });
    return {M.matmul(c_), M.matmul(G_)};
}

Zonotope Zonotope::linear_map(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.inf().shape().size() >= 2 && I.inf().shape().back() == dim(),
               Error::dimension_mismatch, [&] {
                   return "linear_map: the interval matrix needs " + std::to_string(dim()) +
                          " columns (the dimension of Z), got the shape " +
                          shape_text(I.inf().shape());
               });
    const Tensor center = I.center();
    // The center matrix maps Z; the radius adds one axis-aligned generator per dimension.
    return {center.matmul(c_), Tensor::cat_last({center.matmul(G_), aux_widening(*this, I).diag()})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
