// split_into_convex_sets - the convex pieces of a polygon, as CORA's splitIntoConvexSets
//
// The stored pieces (CORALean's Polygon.splitIntoConvexSets): convex, and their union is the
// polygon. CORA merges the triangles of its polyshape greedily, which the boundary
// constructor does too.
//
// Syntax:   std::vector<Polytope> pieces = pgon.split_into_convex_sets();
// Inputs:   -
// Outputs:  pieces - vertex polytopes (k, 2)
// See also: pieces, triangulation
//
// CORA: contSet/@polygon/splitIntoConvexSets.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polytope> Polygon::split_into_convex_sets() const {
    detail::Entry entry;
    return pieces_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
