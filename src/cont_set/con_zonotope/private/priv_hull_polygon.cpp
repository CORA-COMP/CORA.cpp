// priv_hull_polygon - the convex hull of points in the plane as a polygon
//
// Andrew's monotone chain: sorted points, a stack per chain, collinear points dropped.
//
// Syntax:   Polygon hull = priv_hull_polygon(V);
// Inputs:   V - points as columns (2, K), K >= 0
// Outputs:  hull - the corners counter-clockwise; a point or a segment for a flat set; none for K =
// 0 See also: priv_vertex_candidates, ConZonotope::vertices
//
// CORA.cpp only: the convex hull of plane points, which MATLAB's convhull supplies.
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The twice signed area of the turn o -> a -> b.
double aux_cross(const Point &o, const Point &a, const Point &b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/// The points left of the chain a, b, ..., kept in order (one pass of the monotone chain).
Polygon aux_chain(const std::vector<Point> &points) {
    Polygon chain;
    for (const Point &p : points) {
        while (chain.size() >= 2 && aux_cross(chain[chain.size() - 2], chain.back(), p) <= 0.0)
            chain.pop_back();
        chain.push_back(p);
    }
    return chain;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polygon priv_hull_polygon(const Tensor &V) {
    const std::vector<double> x = V.data();
    const std::size_t K = x.size() / 2;
    std::vector<Point> points(K);
    for (std::size_t k = 0; k < K; ++k) points[k] = {x[k], x[K + k]};
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() <= 2) return points;
    // the lower chain from the left to the right, the upper chain back
    Polygon hull = aux_chain(points);
    hull.pop_back();
    std::reverse(points.begin(), points.end());
    Polygon upper = aux_chain(points);
    upper.pop_back();
    hull.insert(hull.end(), upper.begin(), upper.end());
    return hull;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
