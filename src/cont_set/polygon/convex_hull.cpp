// convex_hull - the convex hull of a polygon, as CORA's convHull_ and polytope
//
// The vertex polytope of the vertices of every piece, in their order (CORALean's
// Polygon.convHull, the pieces' vertex lists joined); no piece gives the empty polytope.
//
// Syntax:   Polytope P = pgon.convex_hull();   Polytope P = pgon.polytope();
// Inputs:   -
// Outputs:  P - the convex hull, a vertex polytope (k, 2)
// See also: split_into_convex_sets, Polytope::convex_hull
//
// CORA: contSet/@polygon/convHull_.m
// CORA: contSet/@polygon/polytope.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polygon::convex_hull() const {
    detail::Entry entry;
    if (pieces_.empty()) return Polytope(like_);
    std::vector<Tensor> parts;
    parts.reserve(pieces_.size());
    for (const Polytope &P : pieces_) parts.push_back(P.V());
    return Polytope(Tensor::concat(parts, 0));
}

Polytope Polygon::polytope() const { return convex_hull(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
