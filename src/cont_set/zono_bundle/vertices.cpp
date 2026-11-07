// vertices - the vertices of a zonotope bundle (plane and space)
//
// The vertices of the constrained zonotope of the bundle. CORA intersects the polytopes of the
// members; the set is the same. Not a batch.
//
// Syntax:   std::vector<Polygon> V = bundle.vertices();   auto V3 = bundle.vertices_3d();
// Inputs:   -
// Outputs:  V - the vertices of a planar bundle, counter-clockwise
//           V3 - the vertices of a bundle in space, in no order
// See also: con_zonotope, ConZonotope::vertices
//
// CORA: contSet/@zonoBundle/vertices_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> ZonoBundle::vertices_impl() const { return con_zonotope().vertices(); }

std::vector<std::vector<Point3>> ZonoBundle::vertices_3d_impl() const {
    return con_zonotope().vertices_3d();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
