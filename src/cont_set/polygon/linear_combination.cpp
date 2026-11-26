// linear_combination - the linear combination of two polygons, as CORA's linComb
//
// For every pair of pieces, one of each in this order, the hull of both: their vertex lists
// joined (CORALean's Polygon.linComb, VPolytope.linComb).
//
// Syntax:   Polygon L = pgon.linear_combination(S);
// Inputs:   S - a polygon
// Outputs:  L - the pieces' pairwise hulls
// See also: convex_hull, unite
//
// CORA: contSet/@polygon/linComb.m
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

Polygon Polygon::linear_combination(const Polygon &S) const {
    detail::Entry entry;
    std::vector<Polytope> pieces;
    for (const Polytope &P : pieces_)
        for (const Polytope &Q : S.pieces_) pieces.emplace_back(Tensor::concat({P.V(), Q.V()}, 0));
    return priv_polygon(std::move(pieces), like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
