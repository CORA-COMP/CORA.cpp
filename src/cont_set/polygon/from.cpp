// from - a two-dimensional set as a polygon, as CORA's polygon(S)
//
// The outline of the set (ContSet::vertices, counter-clockwise), cut into convex pieces if
// it is not convex; a polygon is itself.
//
// Syntax:   Polygon pgon = Polygon::from(Z);
// Inputs:   S - a set of two dimensions, bounded, not a batch
// Outputs:  pgon - the polygon of its outline
// See also: polygon, vertices
//
// CORA: contSet/@polygon/polygon.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polygon Polygon::from(const ContSet &S) {
    detail::Entry entry;
    CORA_CHECK(S.dim() == 2, Error::dimension_mismatch, [&] {
        return "from: a polygon is two-dimensional, the set has dimension " +
               std::to_string(S.dim());
    });
    if (S.class_id() == kClassId) return static_cast<const Polygon &>(S);
    const std::vector<Outline> outlines = S.vertices();
    CORA_CHECK(outlines.size() <= 1, Error::not_supported,
               [] { return std::string("from: the set must be a single set, not a batch"); });
    // the outline as a boundary
    const Tensor like = S.interval().inf().zeros_like({0, 2});
    std::vector<Piece> pieces;
    if (!outlines.empty())
        pieces = priv_convex_split(Piece(outlines.front().begin(), outlines.front().end()));
    return priv_polygon_of(pieces, like);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
