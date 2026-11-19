// enclose_points - the smallest box around points, as CORA's interval.enclosePoints
//
// Syntax:   Interval I = Interval::enclose_points(points);
// Inputs:   points - the points as columns, (..., n, N) with N >= 1
// Outputs:  I - [min, max] over the columns, per row; nothing rounds
// See also: convex_hull
//
// CORA: contSet/@interval/enclosePoints.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::enclose_points(const Tensor &points) {
    detail::Entry entry;
    CORA_CHECK(points.ndim() >= 2 && points.size(-2) >= 1, Error::wrong_value, [&] {
        return "enclose_points: give at least one point, as rows (..., N, n), got the shape " +
               shape_text(points.shape());
    });
    return {points.min(-2), points.max(-2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
