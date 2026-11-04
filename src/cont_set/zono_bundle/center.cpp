// center - a point of the bundle near its middle, as CORA's zonoBundle.center
//
// The center of the constrained zonotope of the bundle (a Chebyshev center of its factors); a
// column without entries (n, 0) for an empty bundle. Not a batch if there is more than one member.
//
// Syntax:   Tensor c = bundle.center();
// Inputs:   -
// Outputs:  c - a point of the bundle, a column (n, 1)
// See also: con_zonotope, ConZonotope::center
//
// CORA: contSet/@zonoBundle/center.m
//
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ZonoBundle::center_impl() const { return con_zonotope().center(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
