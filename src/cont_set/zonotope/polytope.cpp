// polytope - the halfspace form of a zonotope, as CORA's zonotope.polytope
//
// The facets of the zonotope: the normals of the cross products of its generators with both signs
// and the support values as offsets, in CORALean's arithmetic and order
// (priv_zonotope_facet_normals); a zonotope without generators is its box, a degenerate one the
// facets of its projection onto the span of the generators and two halfspaces per flat direction
// (priv_zonotope_halfspaces). Not a batch.
//
// Syntax:   Polytope P = Z.polytope();
// Inputs:   -
// Outputs:  P - the same set as {x | C x <= d}, C = [C0; -C0]
// See also: Zonotope::simplex, ConZonotope::polytope, ZonoBundle::polytope
//
// CORA: contSet/@zonotope/polytope.m
// CORA authors: Niklas Kochdumper, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Zonotope::polytope() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("polytope: it supports a single zonotope, not a batch"); });
    const auto [C, d] = priv_zonotope_halfspaces(*this);
    return {C, d};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
