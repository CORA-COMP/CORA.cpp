// mtimes - the linear map M * Z, as CORA's zonotope.mtimes
//
// Syntax:   Zres = Z.mtimes(M);   Zres = Z.mtimes(I);
// Inputs:   M - matrix (..., n, n);  I - interval matrix: an Interval with (..., n, n) bounds
// Outputs:  Zres - M * Z exactly; for an interval matrix, a zonotope enclosing [I] * Z
// See also: plus, linComb, Interval::mtimes

#include "contSet/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// How far the radius of I can move a point of Z along each axis: rad(I) * (|c| + sum_j |G_j|).
Tensor aux_widening(const Zonotope &Z, const Interval &I) {
    const Tensor reach = Z.c.abs() + Z.G.abs().sumLast();
    return I.rad().matmul(reach);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::mtimes(const Tensor &M) const { return {M.matmul(c), M.matmul(G)}; }

Zonotope Zonotope::mtimes(const Interval &I) const {
    const Tensor center = I.center();
    // The center matrix maps Z; the radius adds one axis-aligned generator per dimension.
    return {center.matmul(c), Tensor::catLast({center.matmul(G), aux_widening(*this, I).diag()})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
