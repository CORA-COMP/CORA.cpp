// enclose - a polytope that encloses a polytope and a second set, as CORA's polytope.enclose
//
// Syntax:   Pres = P.enclose(P2);   Pres = P.enclose(M, Pplus);
// Inputs:   P2 - a polytope of the dimension of P
//           M, Pplus - a matrix and a polytope for the
//           second set M P + Pplus
// Outputs:  Pres - the convex hull of P and P2, which is exact because the hull is convex
//           (CORA's enclose calls convHull). The second form needs the image and the Minkowski sum
//           of polytopes, which are not ported yet: Error::not_supported
// See also: convex_hull, Zonotope::enclose
//
// CORA: contSet/@polytope/enclose.m
// CORA authors: Matthias Althoff, Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::enclose(const Polytope &P2) const {
    detail::Entry entry;
    CORA_CHECK(P2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(P2.dim());
    });
    return convex_hull(P2);
}

Polytope Polytope::enclose(const Tensor &, const Polytope &) const {
    detail::Entry entry;
    CORA_FAIL(Error::not_supported, "enclose: the form enclose(M, Pplus) needs the image M P and "
                                    "the sum with Pplus of polytopes, which are not ported yet");
}

Set Polytope::enclose_impl(const ContSet &S2) const {
    if (S2.class_id() == ClassId::polytope) return enclose(static_cast<const Polytope &>(S2));
    priv_noops("enclose", *this, &S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
