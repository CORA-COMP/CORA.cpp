// is_intersecting - whether two polygons meet, as CORA's isIntersecting_
//
// False only if every pair of pieces is strictly separated along the normal (d1, -d0) of a
// difference d of two vertices of either piece (CORALean's Polygon.isDisjoint: the support
// bounds of the two pieces along it). All edges are among the differences, so convex pieces
// that do not meet are always separated; touching pieces meet.
//
// Syntax:   bool b = pgon.is_intersecting(S);
// Inputs:   S - a polygon
//           tol - only 0 (default: unset)
// Outputs:  b - whether the polygons share a point
// See also: contains, intersect
//
// CORA: contSet/@polygon/isIntersecting_.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The normals of the vertex differences of the piece, vertex l before m.
std::vector<Point> aux_normals(const Piece &r) {
    std::vector<Point> out;
    for (const Point &a : r)
        for (const Point &b : r) out.push_back({a[1] - b[1], b[0] - a[0]});
    return out;
}

/// Whether some normal keeps p strictly below q.
bool aux_separated(const Piece &p, const Piece &q) {
    std::vector<Point> normals = aux_normals(p);
    const std::vector<Point> more = aux_normals(q);
    normals.insert(normals.end(), more.begin(), more.end());
    for (const Point &d : normals)
        if (priv_bound_dot(p, d) < -priv_bound_dot(q, {-d[0], -d[1]})) return true;
    return false;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Polygon::is_intersecting(const Polygon &S, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(!tol || *tol == 0, Error::not_supported, [] {
        return std::string("is_intersecting: only tol = 0 is supported, as in CORALean");
    });
    const std::vector<Piece> A = priv_hosts(*this), B = priv_hosts(S);
    for (const Piece &p : A)
        for (const Piece &q : B)
            if (!aux_separated(p, q)) return true;
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
