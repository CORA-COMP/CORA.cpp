// vertices - the vertices of a two-dimensional set, checked, as CORA's contSet.vertices
//
// Syntax:   V = S.vertices();
// Inputs:   S - two-dimensional set
// Outputs:  V - one polygon per batch member, its corners counter-clockwise
// See also: project, Zonotope::vertices_impl
//
// CORA: contSet/@contSet/vertices.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> ContSet::vertices() const {
    detail::Entry entry;
    CORA_CHECK(dim() == 2 && center().shape().back() == 1, Error::not_supported, [&] {
        return "vertices: needs a two-dimensional set of vectors, this one has " +
               std::to_string(dim()) + " dimensions; project it first, e.g. S.project({0, 1})";
    });
    return vertices_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
