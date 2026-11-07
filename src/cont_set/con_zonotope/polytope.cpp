// polytope - the halfspace form of a constrained zonotope, as CORA's conZonotope.polytope
//
// The lifted zonotope [c; -b] + [G; A] x is the set of the points (cz point, A x - b); the facets
// of it, restricted to the first n coordinates, bound the set. CORA takes the vertices in the
// plane; the facets of the lifted zonotope are exact in every dimension. Not a batch.
//
// Syntax:   Polytope P = cz.polytope();
// Inputs:   -
// Outputs:  P - the same set as {x | C x <= d}
// See also: zono_bundle, Polytope::con_zonotope
//
// CORA: contSet/@conZonotope/polytope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope ConZonotope::polytope() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("polytope: it supports a single set, not a batch"); });
    // the facets of the lifted zonotope, cut at 0 in the constraint coordinates
    const Zonotope lifted(Tensor::concat({c_, b_ * -1.0}, -2), Tensor::concat({G_, A_}, -2));
    const auto [C, d] = priv_zonotope_halfspaces(lifted);
    return {C.slice(-1, 0, dim()), d};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
