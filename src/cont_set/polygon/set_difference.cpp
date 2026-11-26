// set_difference - the difference of two polygons, as CORA's subtract and mldivide
//
// The pieces made disjoint, each minus every piece of S in turn, each step one part per edge
// of the subtracted hull (priv_minus_piece): disjoint convex parts, the closure of the difference.
// CORALean cuts the pieces to boxes instead.
//
// Syntax:   Polygon R = pgon.set_difference(S);
// Inputs:   S - a polygon
// Outputs:  R - the points of this polygon not inside S
// See also: intersect, priv_minus_piece
//
// CORA: contSet/@polygon/subtract.m
// CORA: contSet/@polygon/mldivide.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polygon Polygon::set_difference(const Polygon &S) const {
    detail::Entry entry;
    std::vector<Piece> subtrahends;
    for (const Piece &q : priv_hosts(S)) subtrahends.push_back(priv_hull(q));
    std::vector<Piece> parts;
    // the disjoint parts of this polygon, each cut by every piece of S
    for (const Piece &p : priv_disjoint(priv_hosts(*this))) {
        std::vector<Piece> mine = {p};
        // cut by every piece of S
        for (const Piece &q : subtrahends) {
            std::vector<Piece> next;
            for (const Piece &m : mine)
                for (Piece &d : priv_minus_piece(m, q)) next.push_back(std::move(d));
            mine = std::move(next);
        }
        parts.insert(parts.end(), mine.begin(), mine.end());
    }
    return priv_polygon_of(parts, like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
