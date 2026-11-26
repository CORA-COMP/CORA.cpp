// triangulation - triangles that cover a polygon without overlap, as CORA's triangulation
//
// The pieces made disjoint, each part cut into the fan of its first vertex. CORALean's
// triangulation is every vertex triple of every piece, a cover with overlaps.
//
// Syntax:   std::vector<Polytope> T = pgon.triangulation();
// Inputs:   -
// Outputs:  T - vertex polytopes (3, 2), counter-clockwise
// See also: split_into_convex_sets, volume
//
// CORA: contSet/@polygon/triangulation.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polytope> Polygon::triangulation() const {
    detail::Entry entry;
    std::vector<Polytope> triangles;
    for (const Piece &part : priv_disjoint(priv_hosts(*this)))
        for (std::size_t i = 1; i + 1 < part.size(); ++i)
            triangles.push_back(priv_piece({part[0], part[i], part[i + 1]}, like_));
    return triangles;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
