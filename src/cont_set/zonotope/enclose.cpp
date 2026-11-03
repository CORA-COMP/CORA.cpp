// enclose - a zonotope that encloses two zonotopes, as CORA's zonotope.enclose
//
// Syntax:   Zres = Z.enclose(Z2);
// Inputs:   Z2 - zonotope of the same dimension
// Outputs:  Zres - encloses conv(Z, Z2): center (c+c2)/2; the generators in common (the first ones
//           of the zonotope with fewer) give (G+G2)/2 and (G-G2)/2, then come (c-c2)/2 and the
//           extra generators of the other one
// See also: linear_combination, convex_hull
//
// CORA: contSet/@zonotope/enclose.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::enclose(const Zonotope &Z2) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    const Tensor &G2 = Z2.generators();
    const int64_t m = G_.shape().back(), m2 = G2.shape().back();

    // the one with more generators is cut where the other ends
    const bool second_shorter = m2 <= m;
    const Tensor &longer = second_shorter ? G_ : G2, &shorter = second_shorter ? G2 : G_;
    const int64_t common = second_shorter ? m2 : m, total = second_shorter ? m : m2;
    const Tensor half_distance = second_shorter ? (c_ - Z2.center()) / 2.0
                                                : (Z2.center() - c_) / 2.0;
    const Tensor cut = longer.slice(-1, 0, common);
    std::vector<Tensor> parts{(cut + shorter) / 2.0, half_distance, (cut - shorter) / 2.0};
    if (total > common) parts.push_back(longer.slice(-1, common, total));
    return {(c_ + Z2.center()) / 2.0, Tensor::concat(parts, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
