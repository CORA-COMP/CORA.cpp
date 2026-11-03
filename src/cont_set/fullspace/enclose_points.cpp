// enclose_points - the whole space of the dimension of points, as CORA's fullspace.enclosePoints
//
// Syntax:   Fullspace F = Fullspace::enclose_points(points);
// Inputs:   points - the points as columns, (..., n, N)
// Outputs:  F - R^n, which contains the points
// See also: Interval::enclose_points
//
// CORA: contSet/@fullspace/enclosePoints.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::enclose_points(const Tensor &points) {
    detail::Entry entry;
    CORA_CHECK(points.ndim() >= 2, Error::wrong_value, [&] {
        return "enclose_points: give the points as columns (..., n, N), got the shape " +
               shape_text(points.shape());
    });
    return Fullspace(points.size(-2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
