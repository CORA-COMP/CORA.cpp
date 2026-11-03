// enclose_points - the smallest box around points, as CORA's interval.enclosePoints
//
// Syntax:   Interval I = Interval::enclose_points(points);
// Inputs:   points - the points as columns, (..., n, N) with N >= 1
// Outputs:  I - [min, max] over the columns, per row; nothing rounds
// See also: convex_hull
//
// CORA: contSet/@interval/enclosePoints.m

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::enclose_points(const Tensor &points) {
    detail::Entry entry;
    CORA_CHECK(points.ndim() >= 2 && points.size(-1) >= 1, Error::wrong_value, [&] {
        return "enclose_points: give at least one point, as columns (..., n, N), got the shape " +
               shape_text(points.shape());
    });
    return {points.min(-1, true), points.max(-1, true)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
