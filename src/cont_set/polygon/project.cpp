// project - a polygon at two of its coordinates, or one, as CORA's project
//
// Two coordinates: every piece read at dims (CORALean's Polygon.project, Polytope::project),
// a swap or the same set. One coordinate: the interval hull of that coordinate, as CORA.
//
// Syntax:   Set R = pgon.project({1, 0});
// Inputs:   dims - one or two coordinates (0-based)
// Outputs:  R - a polygon, or an interval for one coordinate
// See also: interval, linear_map
//
// CORA: contSet/@polygon/project.m
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

Set Polygon::project_impl(const std::vector<int64_t> &dims) const {
    if (dims.size() == 1) return interval_impl().project(dims);
    if (dims.size() != 2)
        CORA_FAIL(Error::dimension_mismatch,
                  "project: a polygon keeps one or two of its coordinates");
    std::vector<Polytope> pieces;
    pieces.reserve(pieces_.size());
    for (const Polytope &P : pieces_) pieces.push_back(P.project(dims));
    return priv_polygon(std::move(pieces), like_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
