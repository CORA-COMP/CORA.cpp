// unite - the union of two polygons, as CORA's or
//
// The pieces of both, nothing computed (CORALean's Polygon.or).
//
// Syntax:   Polygon U = pgon.unite(S);
// Inputs:   S - a polygon
// Outputs:  U - the union
// See also: intersect, set_difference
//
// CORA: contSet/@polygon/or.m
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

Polygon Polygon::unite(const Polygon &S) const {
    detail::Entry entry;
    std::vector<Polytope> pieces = pieces_;
    pieces.insert(pieces.end(), S.pieces_.begin(), S.pieces_.end());
    return priv_polygon(std::move(pieces), like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
