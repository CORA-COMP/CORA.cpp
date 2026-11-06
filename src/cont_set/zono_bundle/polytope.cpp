// polytope - the halfspace form of a zonotope bundle, as CORA's zonoBundle.polytope
//
// The intersection of the polytopes of the members, each the facets of the zonotope (CORA's
// polytope(Z)); the constraints of the members are stacked. Not a batch.
//
// Syntax:   Polytope P = bundle.polytope();
// Inputs:   -
// Outputs:  P - the same set as {x | A x <= b}
// See also: con_zonotope, Polytope::intersect
//
// CORA: contSet/@zonoBundle/polytope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The polytope of one member: its facets C x <= d.
Polytope aux_member(const Zonotope &Z) {
    const auto [C, d] = priv_zonotope_halfspaces(Z);
    return {C, d};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope ZonoBundle::polytope() const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_batch_refusal(Z_.front().center(), "polytope");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    Polytope P = aux_member(Z_.front());
    for (std::size_t i = 1; i < Z_.size(); ++i) P = P.intersect(aux_member(Z_[i]));
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
