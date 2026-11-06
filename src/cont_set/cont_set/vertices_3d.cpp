// vertices_3d - the corners of a three-dimensional set, as CORA's contSet vertices in 3-D
//
// Syntax:   V = S.vertices_3d();
// Inputs:   -
// Outputs:  V - one list of points per batch member
// See also: vertices, project
//
// CORA: contSet/@contSet/vertices.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> ContSet::vertices_3d() const {
    detail::Entry entry;
    CORA_CHECK(dim() == 3 && center().shape().back() <= 1, Error::not_supported, [&] {
        return "vertices_3d: needs a three-dimensional set of vectors, this one has " +
               std::to_string(dim()) + " dimensions; project it first, e.g. S.project({0, 1, 2})";
    });
    return vertices_3d_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
