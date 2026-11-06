// unite - an enclosure of the union of a zonotope bundle with a set, as CORA's zonoBundle.or
//
// The constrained zonotope of the bundle is united with the set (ConZonotope::unite) and the result
// is converted back to a bundle. Not a batch.
//
// Syntax:   ZonoBundle R = bundle.unite(S);
// Inputs:   S - a bundle, a zonotope or a box
// Outputs:  R - a bundle that contains both sets
// See also: con_zonotope, ConZonotope::unite, ConZonotope::zono_bundle
//
// CORA: contSet/@zonoBundle/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::unite(const ZonoBundle &S) const {
    detail::Entry entry;
    // the union is taken on the constrained zonotopes and converted back
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "unite: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    return con_zonotope().unite(S.con_zonotope()).zono_bundle();
}

ZonoBundle ZonoBundle::unite(const Zonotope &Z) const {
    detail::Entry entry;
    CORA_CHECK(Z.dim() == dim(), Error::dimension_mismatch, [&] {
        return "unite: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z.dim());
    });
    // a zonotope is its constrained zonotope without constraints
    return con_zonotope().unite(Z).zono_bundle();
}

ZonoBundle ZonoBundle::unite(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim(), Error::dimension_mismatch, [&] {
        return "unite: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(I.dim());
    });
    // the box is a zonotope as well
    return con_zonotope().unite(I).zono_bundle();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
