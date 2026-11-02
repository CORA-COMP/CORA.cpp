// linear_combination - a zonotope enclosing the segments between two zonotopes, as CORA's linComb
//
// Syntax:   Zres = Z.linear_combination(Z2);
// Inputs:   Z2 - zonotope Z2 = M*Z + t: it shares its generator factors with Z (m generators)
// Outputs:  Zres - encloses conv(Z, Z2) with 2m + 1 generators:
//           center (c+c2)/2; generators (G+G2)/2, (G-G2)/2 and (c-c2)/2
// See also: linear_map, plus
//
// CORA: contSet/@contSet/linComb.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::linear_combination(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(Z2.center().shape() == c_.shape() && Z2.generators().shape() == G_.shape(),
               Error::dimension_mismatch, [&] {
                   return "linear_combination: Z2 must have the shapes of Z (center " +
                          shape_text(c_.shape()) + ", generators " + shape_text(G_.shape()) +
                          "), got " + shape_text(Z2.center().shape()) + " and " +
                          shape_text(Z2.generators().shape());
               });
    const Tensor mean = (G_ + Z2.generators()) * 0.5;  // the shared part of matching generators
    const Tensor spread = (G_ - Z2.generators()) * 0.5;  // how far apart they are
    const Tensor shift = (c_ - Z2.center()) * 0.5;  // the distance of the centers
    return {(c_ + Z2.center()) * 0.5, Tensor::cat_last({mean, spread, shift})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
