// vertices - the outlines of a polygon, as CORA's vertices_
//
// One counter-clockwise loop per piece, the extreme vertices of the piece (what plot unites);
// the vertices as stored are pieces(). CORA returns the polyshape boundary. A polygon is
// two-dimensional, so vertices_3d is refused.
//
// Syntax:   std::vector<Outline> loops = pgon.vertices();
// Inputs:   -
// Outputs:  loops - the outline of every piece
// See also: pieces, plot
//
// CORA: contSet/@polygon/vertices_.m
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

std::vector<Outline> Polygon::vertices_impl() const {
    std::vector<Outline> out;
    for (const Piece &p : priv_hosts(*this)) {
        const Piece hull = priv_hull(p);
        if (!hull.empty()) out.emplace_back(hull.begin(), hull.end());
    }
    return out;
}

std::vector<std::vector<Point3>> Polygon::vertices_3d_impl() const {
    CORA_FAIL(Error::not_supported, "vertices_3d: a polygon is two-dimensional, use vertices");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
