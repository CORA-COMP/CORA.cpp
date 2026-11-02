// plus - the Minkowski sum, as CORA's zonotope.plus
//
// Syntax:   Zres = Z.plus(Z2);
// Inputs:   Z2 - zonotope of the same dimension
// Outputs:  Zres - Z + Z2: center c + c2, the generators of both
// See also: linear_map, linear_combination
//
// CORA: contSet/@zonotope/plus.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    return {c_ + Z2.center(), Tensor::concat({G_, Z2.generators()}, -1)};
}

Zonotope Zonotope::plus(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim() && I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "plus: the interval must be a box of the dimension " + std::to_string(dim()) +
               " of the zonotope, got bounds of the shape " + shape_text(I.inf().shape());
    });
    // the box as a zonotope: its center, and its radius as one generator per axis
    return {c_ + I.center(), Tensor::concat({G_, I.rad().diag()}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
