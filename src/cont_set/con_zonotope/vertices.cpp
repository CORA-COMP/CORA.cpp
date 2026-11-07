// vertices - the vertices of a constrained zonotope (plane and space)
//
// Without constraints the zonotope's. With constraints the vertex candidates
// (priv_vertex_candidates) are hulled: a polygon in the plane, the corners of the hull faces in
// space. The empty set has none. CORA's template method is not ported. A set with constraints is
// not a batch.
//
// Syntax:   std::vector<Polygon> V = cz.vertices();   auto V3 = cz.vertices_3d();
// Inputs:   -
// Outputs:  V - the vertices of a planar set, counter-clockwise
//           V3 - the vertices of a set in space, in no order
// See also: priv_vertex_candidates, priv_hull_polygon, Zonotope::vertices
//
// CORA: contSet/@conZonotope/vertices_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/plot/geometry.h"
#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The columns of the (3, K) points as points of space.
std::vector<Point3> aux_points(const Tensor &V) {
    const std::vector<double> x = V.data();
    const std::size_t K = x.size() / 3;
    std::vector<Point3> points(K);
    for (std::size_t k = 0; k < K; ++k) points[k] = {x[k], x[K + k], x[2 * K + k]};
    return points;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> ConZonotope::vertices_impl() const {
    if (A_.shape()[A_.ndim() - 2] == 0) return Zonotope(c_, G_).vertices();
    return {priv_hull_polygon(priv_vertex_candidates(*this))};
}

std::vector<std::vector<Point3>> ConZonotope::vertices_3d_impl() const {
    if (A_.shape()[A_.ndim() - 2] == 0) return Zonotope(c_, G_).vertices_3d();
    std::vector<Point3> corners;
    // the corners of the faces of the hull of the candidates are its vertices, each listed once
    for (const std::vector<Point3> &face : hull_faces(aux_points(priv_vertex_candidates(*this))))
        corners.insert(corners.end(), face.begin(), face.end());
    std::sort(corners.begin(), corners.end());
    corners.erase(std::unique(corners.begin(), corners.end()), corners.end());
    return {corners};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
