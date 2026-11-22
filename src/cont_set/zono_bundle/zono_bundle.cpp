// zono_bundle - the constructor of a zonotope bundle, checked, as CORA's zonoBundle
//
// CORA also takes an empty list (a bundle of dimension 0 that it treats as the empty set); the
// intersection of no sets is the whole space, so CORA.cpp refuses it.
//
// Syntax:   ZonoBundle bundle({Z1, Z2});
// Inputs:   zonotopes - a nonempty list of zonotopes of one dimension and one batch shape
// Outputs:  bundle - the intersection of the zonotopes
// See also: empty, Zonotope::Zonotope
// References:
//    [1] M. Althoff, B. H. Krogh. "Zonotope bundles for the efficient computation of reachable
//        sets", CDC 2011.
//
// CORA: contSet/@zonoBundle/zonoBundle.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle::ZonoBundle(std::vector<Zonotope> zonotopes)
    : ContSet(kClassId), Z_(std::move(zonotopes)) {
    detail::Entry entry;
    const detail::Refusal refusal = priv_members_refusal(Z_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

ZonoBundle::ZonoBundle(const ConZonotope &cz) : ZonoBundle(cz.zono_bundle()) {}

ZonoBundle::ZonoBundle(const Polytope &P) : ZonoBundle(P.zono_bundle()) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
