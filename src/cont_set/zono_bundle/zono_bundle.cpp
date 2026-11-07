// zono_bundle - the constructor of a zonotope bundle, checked, as CORA's zonoBundle
//
// CORA also takes an empty list (a bundle of dimension 0 that it treats as the empty set); the
// intersection of no sets is the whole space, so CORA.cpp refuses it.
//
// Syntax:   ZonoBundle bundle({Z1, Z2});
// Inputs:   zonotopes - a nonempty list of zonotopes of one dimension and one batch shape
// Outputs:  bundle - the intersection of the zonotopes
// See also: empty, Zonotope::Zonotope
//
// CORA: contSet/@zonoBundle/zonoBundle.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle::ZonoBundle(std::vector<Zonotope> zonotopes)
    : ContSet(kClassId), Z_(std::move(zonotopes)) {
    detail::Entry entry;
    const detail::Refusal refusal = priv_members_refusal(Z_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
