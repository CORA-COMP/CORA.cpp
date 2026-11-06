// vertices_3d - points on the surface of a three-dimensional capsule, for plots
//
// The points of two spheres of 200 Fibonacci points around c + g and c - g; their convex hull is
// the capsule up to the spacing. CORA plots the hull of two zonotopes that enclose the spheres.
//
// Syntax:   V = C.vertices_3d();
// Inputs:   -
// Outputs:  V - the points, one list per batch member
// See also: vertices
//
// CORA: contSet/@capsule/plot3D.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"
#include "global/sphere.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Capsule::vertices_3d_impl() const {
    const std::vector<double> cs = c_.data(), gs = g_.data(), rs = r_.data();
    const std::vector<Point3> sphere = unit_sphere_points(200);
    std::vector<std::vector<Point3>> out;
    for (std::size_t b = 0; b < rs.size(); ++b) {
        std::vector<Point3> points;
        for (const double sign : {1.0, -1.0})
            for (const Point3 &u : sphere)
                points.push_back({cs[3 * b] + sign * gs[3 * b] + rs[b] * u[0],
                                  cs[3 * b + 1] + sign * gs[3 * b + 1] + rs[b] * u[1],
                                  cs[3 * b + 2] + sign * gs[3 * b + 2] + rs[b] * u[2]});
        out.push_back(points);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
