// linear_map - the image of a polygon under a matrix, as CORA's mtimes
//
// Every piece mapped by M (2, 2) (CORALean's Polygon.mtimes): each vertex row by
// Polytope::linear_map. A matrix of another number of rows leaves the plane: the pieces as
// polytopes would be the image, which a polygon cannot hold, so it is refused. times (diag(v)
// pgon) is ContSet's, through this map: CORALean's Polygon.times.
//
// Syntax:   Set R = pgon.linear_map(M);   Set R = M * pgon;
// Inputs:   M - the matrix (2, 2)
// Outputs:  R - the polygon M pgon
// See also: times, Polytope::linear_map
//
// CORA: contSet/@polygon/mtimes.m
// CORA: contSet/@polygon/times.m
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

Set Polygon::linear_map_impl(const Tensor &M) const {
    if (M.ndim() != 2 || M.size(0) != 2)
        CORA_FAIL(
            Error::dimension_mismatch,
            "linear_map: a polygon maps by a matrix (2, 2) only, so that the image is a polygon");
    std::vector<Polytope> pieces;
    pieces.reserve(pieces_.size());
    for (const Polytope &P : pieces_) pieces.push_back(P.linear_map(M));
    return priv_polygon(std::move(pieces), like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
