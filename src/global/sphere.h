// sphere - points spread evenly on the unit sphere, shared by the 3D vertices of round sets
//
// Syntax:   std::vector<Point3> p = unit_sphere_points(200);
// Inputs:   count - the number of points
// Outputs:  p - the points of the Fibonacci lattice: equal areas, golden-angle steps in longitude
// See also: Ellipsoid::vertices_3d, Capsule::vertices_3d
//
// CORA.cpp only: CORA places the points with eq_point_set (equal-area partitions), which needs
//                the EQSP toolbox
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

#include <cmath>
#include <numbers>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The first `count` points of the Fibonacci lattice on the unit sphere.
inline std::vector<Point3> unit_sphere_points(int count) {
    const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
    std::vector<Point3> points;
    points.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        // z runs from near 1 to near -1 in equal steps, which gives equal areas
        const double z = 1.0 - 2.0 * (i + 0.5) / count, ring = std::sqrt(1.0 - z * z);
        points.push_back({ring * std::cos(golden * i), ring * std::sin(golden * i), z});
    }
    return points;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
