// enclose_points - a capsule that contains a cloud of points, as CORA's capsule.enclosePoints
//
// Syntax:   Capsule C = Capsule::enclose_points(points);
// Inputs:   points - the points as columns (..., n, N)
// Outputs:  C - center the mean c; generator g the vector from c to the farthest point (the first
//           of them); radius the largest distance of a point from the segment c - g to c + g.
//           No points give the empty capsule, one point a capsule that is that point.
//           CORA's radius is the largest length of the projection onto the axis, which is |g|, far
//           more; for n = 1 its generator is max(points) - c, which misses the minimum
//           (docs/cora-differences.md)
// See also: Interval::enclose_points, Zonotope::enclose_points
//
// CORA: contSet/@capsule/enclosePoints.m

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::enclose_points(const Tensor &points) {
    detail::Entry entry;
    CORA_CHECK(points.defined() && points.ndim() >= 2, Error::wrong_value, [&] {
        return std::string("enclose_points: the points must be a matrix (..., n, N) of columns");
    });
    std::vector<int64_t> column_shape = points.shape(), radius_shape = column_shape;
    column_shape.back() = 1;
    radius_shape.back() = 1;
    radius_shape[radius_shape.size() - 2] = 1;
    const int64_t N = points.size(-1);
    if (N == 0)
        return {Unchecked{}, points.zeros_like(column_shape), points.zeros_like(column_shape),
                points.full_like(radius_shape, -1.0)};
    const Tensor c = points.sum(-1, true) / double(N);
    const Tensor d = points.subtract(c);
    // the first point of the largest squared distance from the mean
    const Tensor length2 = d.multiply(d).sum(-2, true);
    const Tensor farthest = length2.argmax(-1, true).broadcast_to(column_shape);
    const Tensor g = d.take_along_axis(farthest, -1);
    const Tensor t = priv_axis_parameter(g, d);
    const Tensor r = priv_norm(d.subtract(t.multiply(g))).max(-1, true);
    return {Unchecked{}, c, g, r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
