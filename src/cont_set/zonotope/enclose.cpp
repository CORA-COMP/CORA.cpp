// enclose - a zonotope that encloses two zonotopes, as CORA's zonotope.enclose
//
// The generators in common (the first ones of the zonotope with fewer) give (G+G2)/2 and
// (G-G2)/2, then come (c-c2)/2 and the extra generators of the other one.
//
// Syntax:   Zres = Z.enclose(Z2);   Zres = Z.enclose(M, Zplus);
// Inputs:   Z2 - zonotope of the same dimension; the second form takes M * Z + Zplus for Z2
// Outputs:  Zres - encloses conv(Z, Z2), with center (c+c2)/2
// See also: linear_combination, convex_hull
//
// CORA: contSet/@zonotope/enclose.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
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
    CORA_CHECK(elem_rank_ == 1 && Z2.elem_rank() == 1, Error::not_supported,
               [] { return std::string("enclose: not for a matrix zonotope"); });
    const Tensor &G2 = Z2.generators();
    const int64_t m = G_.size(-2), m2 = G2.size(-2);

    // the one with more generators is cut where the other ends
    const bool second_shorter = m2 <= m;
    const Tensor &longer = second_shorter ? G_ : G2, &shorter = second_shorter ? G2 : G_;
    const int64_t common = second_shorter ? m2 : m, total = second_shorter ? m : m2;
    const Tensor half_distance =
        (second_shorter ? (c_ - Z2.center()) / 2.0 : (Z2.center() - c_) / 2.0).expand_dims(-2);
    const Tensor cut = longer.slice(-2, 0, common);
    std::vector<Tensor> parts{(cut + shorter) / 2.0, half_distance, (cut - shorter) / 2.0};
    if (total > common) parts.push_back(longer.slice(-2, common, total));
    return {(c_ + Z2.center()) / 2.0, Tensor::concat(parts, -2)};
}

Zonotope Zonotope::enclose(const Tensor &M, const Zonotope &Zplus) const {
    detail::Entry entry;
    // the second zonotope of CORA's three-argument form must lie in the space of the first
    CORA_CHECK(M.size(-2) == dim() && M.size(-1) == dim() && Zplus.dim() == dim(),
               Error::dimension_mismatch, [&] {
                   return "enclose: M must be square of the dimension " + std::to_string(dim()) +
                          " of the zonotope, and Zplus of the same dimension";
               });
    return enclose(linear_map(M) + Zplus);
}

Set Zonotope::enclose_impl(const ContSet &S2) const {
    if (S2.class_id() != ClassId::zonotope) priv_noops("enclose", *this, &S2);
    return enclose(static_cast<const Zonotope &>(S2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
