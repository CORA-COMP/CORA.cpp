// vertices_3d - points on the surface of a three-dimensional ellipsoid, for plots
//
// The image of 200 Fibonacci points of the unit sphere under q + G; their convex hull is the
// ellipsoid up to the spacing. CORA plots an enclosing zonotope of 100 generators.
//
// Syntax:   V = E.vertices_3d();
// Inputs:   -
// Outputs:  V - the points, one list per batch member
// See also: vertices
//
// CORA: contSet/@ellipsoid/plot3D.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"
#include "global/sphere.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Ellipsoid::vertices_3d_impl() const {
    if (priv_is_empty(*this)) CORA_FAIL(Error::empty_set, "vertices_3d: the ellipsoid is empty");
    const Tensor G = generators().matmul(Q_.svd()[0].matrix_transpose());
    const std::vector<double> qs = q_.data(), Gs = G.data();
    const std::vector<Point3> sphere = unit_sphere_points(200);
    std::vector<std::vector<Point3>> out;
    for (std::size_t b = 0; b < qs.size() / 3; ++b) {
        const double *g = &Gs[9 * b];
        std::vector<Point3> points;
        for (const Point3 &u : sphere) {
            Point3 p = {qs[3 * b], qs[3 * b + 1], qs[3 * b + 2]};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    p[static_cast<std::size_t>(i)] += g[3 * i + j] * u[static_cast<std::size_t>(j)];
            points.push_back(p);
        }
        out.push_back(points);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
