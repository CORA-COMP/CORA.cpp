// intersect - the intersection of two polygons, as CORA's and_
//
// Every pair of pieces clipped (the hull of one by the edge halfplanes of the hull of the
// other, Sutherland-Hodgman); a part without area is dropped, as polyshape drops it. Exact up
// to the rounding of the crossing points. CORALean intersects the pieces' boxes instead.
//
// Syntax:   Polygon R = pgon.intersect(S);   Set R = pgon & S;
// Inputs:   S - a polygon
// Outputs:  R - the intersection
// See also: set_difference, unite, priv_clip
//
// CORA: contSet/@polygon/and_.m
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

Polygon Polygon::intersect(const Polygon &S) const {
    detail::Entry entry;
    std::vector<Piece> parts;
    for (const Piece &p : priv_hosts(*this))
        for (const Piece &q : priv_hosts(S)) {
            Piece part = priv_hull(p);
            const Piece hull = priv_hull(q);
            if (hull.size() < 3) continue;
            // inside every edge of q's hull
            for (std::size_t i = 0; i < hull.size() && !part.empty(); ++i) {
                const Point a = hull[i], b = hull[(i + 1) % hull.size()];
                const Point n = {b[1] - a[1], a[0] - b[0]};
                part = priv_clip(part, n, n[0] * a[0] + n[1] * a[1]);
            }
            if (part.size() >= 3) parts.push_back(std::move(part));
        }
    return priv_polygon_of(parts, like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
