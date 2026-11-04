// center - a point of the bundle near its middle, as CORA's zonoBundle.center
//
// Syntax:   Tensor c = zB.center();
// Inputs:   -
// Outputs:  c - the center of the constrained zonotope of the bundle (a Chebyshev center of its
//           factors, ConZonotope::center); a column without entries (n, 0) for an empty bundle.
//           Not a batch if the bundle has more than one member
// See also: con_zonotope, ConZonotope::center
//
// CORA: contSet/@zonoBundle/center.m

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ZonoBundle::center_impl() const { return con_zonotope().center(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
