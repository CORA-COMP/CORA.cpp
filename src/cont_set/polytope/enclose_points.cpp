// enclose_points - the polytope spanned by a cloud of points, as CORA's polytope.enclosePoints
//
// Syntax:   Polytope P = Polytope::enclose_points(points);
// Inputs:   points - the points as columns (n, N), finite unless n = 1
// Outputs:  P - the vertex polytope of the extreme points of the cloud in their order (of equal
//           points the first); a degenerate cloud (flat, a segment, a point) works as well. CORA
//           reduces the cloud to the span of its columns by a QR factorization and takes the
//           facets of its convex hull with qhull, which fails for a flat cloud that does not
//           span the origin; here a point is extreme if it is no convex combination of the others
//           (a linear program each). Not a batch
// See also: convex_hull, Interval::enclose_points
//
// CORA: contSet/@polytope/enclosePoints.m
//
// CORA authors: Niklas Kochdumper, Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::enclose_points(const Tensor &points) {
    detail::Entry entry;
    CORA_CHECK(points.defined() && points.ndim() == 2, Error::wrong_value, [&] {
        return "enclose_points: the points must be a matrix (n, N) of columns, got the shape " +
               (points.defined() ? shape_text(points.shape()) : std::string("of an empty tensor"));
    });
    const Rows cloud = priv_host_points(points);
    if (cloud.size() <= 1) return Polytope(points);
    return Polytope(points.take(priv_extreme_points(cloud), -1));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
