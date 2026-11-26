// center - the centroid of a polygon, as CORA's center
//
// The centroid of the region: the area-weighted centroids of the disjoint parts (CORA's
// centroid of the polyshape). Without area, the mean of the vertices; NaN for no vertex.
// CORALean has no float centroid.
//
// Syntax:   Tensor c = pgon.center();
// Inputs:   -
// Outputs:  c - the centroid (2)
// See also: volume, interval
//
// CORA: contSet/@polygon/center.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The mean of all vertices, NaN for none.
Point aux_mean(const std::vector<Piece> &pieces) {
    double n = 0;
    Point sum = {0, 0};
    for (const Piece &p : pieces)
        for (const Point &x : p) {
            sum = {sum[0] + x[0], sum[1] + x[1]};
            n += 1;
        }
    if (n == 0)
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    return {sum[0] / n, sum[1] / n};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Polygon::center_impl() const {
    const std::vector<Piece> hosts = priv_hosts(*this);
    double area = 0;
    Point moment = {0, 0};
    // the centroid of a loop: the sums of (x_i + x_{i+1}) (x_i y_{i+1} - x_{i+1} y_i) over 6 A
    for (const Piece &part : priv_disjoint(hosts))
        for (std::size_t i = 0; i < part.size(); ++i) {
            const Point a = part[i], b = part[(i + 1) % part.size()];
            const double w = a[0] * b[1] - b[0] * a[1];
            area += w / 2;
            moment = {moment[0] + (a[0] + b[0]) * w / 6, moment[1] + (a[1] + b[1]) * w / 6};
        }
    const Point c = area > 0 ? Point{moment[0] / area, moment[1] / area} : aux_mean(hosts);
    return Tensor::like(like_, {c[0], c[1]}, {2});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
