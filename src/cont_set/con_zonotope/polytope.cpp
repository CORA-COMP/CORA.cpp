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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope ConZonotope::polytope() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 1, Error::not_supported,
               [] { return std::string("polytope: it supports a single set, not a batch"); });
    // the facets of the lifted zonotope, cut at 0 in the constraint coordinates
    const Zonotope lifted = Zonotope(
        (Tensor::concat({c_.expand_dims(-1), priv_b_column(*this) * -1.0}, -2)).squeeze(-1),
        (Tensor::concat({G_.matrix_transpose(), A()}, -2)).matrix_transpose());
    const auto [C, d] = priv_zonotope_halfspaces(lifted);
    return {C.slice(-1, 0, dim()), d.squeeze(-1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
