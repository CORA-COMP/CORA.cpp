// convex_hull - a polynomial zonotope that contains the convex hull, as CORA's convHull_
//
// With no second set CORA returns linComb(pZ, pZ), the segments between two points of the set (no
// convex hull in general, docs/cora-differences.md); so does this. With a second set the
// dependent parts are paired with themselves by linear_combination, those chord sets with each
// other, and the independent generators are those of the combination of the two zero-centered
// zonotopes of the independent blocks, as CORALean's convHull: the pairing, then the column of the
// half-difference of their centers, which is zero.
//
// Syntax:   S = pZ.convex_hull();   S = pZ.convex_hull(S2);
// Inputs:   S2 - polynomial zonotope, zonotope or box of the dimension of pZ
// Outputs:  S - a polynomial zonotope that contains both sets
// See also: linear_combination, enclose
//
// CORA: contSet/@polyZonotope/convHull_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::convex_hull() const {
    detail::Entry entry;
    return linear_combination(*this);
}

PolyZonotope PolyZonotope::convex_hull(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "convex_hull: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // the chords of the dependent parts with themselves, then with each other
    const PolyZonotope hull =
        no_indep()
            .linear_combination(no_indep())
            .linear_combination(S.no_indep().linear_combination(S.no_indep()));
    // the independent generators: the pairing of the two blocks and the (zero) center column
    const Tensor pairs = priv_pair_independent(GI_, S.GI_);
    const Tensor GI = Tensor::concat({pairs, c_.zeros_like({1, dim()})}, -2);
    return {hull.center(), hull.generators(), GI, hull.exponents(), hull.ids(), detail::Verbatim{}};
}

PolyZonotope PolyZonotope::convex_hull(const Zonotope &Z) const {
    detail::Entry entry;
    return convex_hull(PolyZonotope(Z));
}

PolyZonotope PolyZonotope::convex_hull(const Interval &I) const {
    detail::Entry entry;
    return convex_hull(PolyZonotope(I));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
