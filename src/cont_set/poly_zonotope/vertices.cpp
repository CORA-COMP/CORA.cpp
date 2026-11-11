// vertices - the vertices of a polynomial zonotope: CORA has none
//
// CORA's polyZonotope has no vertices_ (its polygon method plots an outline); the sets that
// plot over-approximate through zonotope().
//
// Syntax:   V = pZ.vertices();   V = pZ.vertices_3d();
// Inputs:   -
// Outputs:  an Error::not_supported
// See also: zonotope
//
// CORA: contSet/@polyZonotope/polygon.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> PolyZonotope::vertices_impl() const {
    CORA_FAIL(
        Error::not_supported,
        "vertices: a polynomial zonotope has no vertices; use zonotope() to plot an enclosure");
}

std::vector<std::vector<Point3>> PolyZonotope::vertices_3d_impl() const {
    CORA_FAIL(
        Error::not_supported,
        "vertices_3d: a polynomial zonotope has no vertices; use zonotope() for an enclosure");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
