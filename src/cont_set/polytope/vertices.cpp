// vertices - the vertices of a polytope in the plane and in space, as CORA's polytope.vertices_
//
// The extreme points, found by the vertex form or by enumerating the rows. The plane gets them in
// counter-clockwise order, space in no order. Unbounded sets have no vertex list
// (Error::special_error); the empty set none. Not a batch.
//
// Syntax:   std::vector<Polygon> V = P.vertices();   auto V = P.vertices_3d();
// Inputs:   -
// Outputs:  V - the corners of the plane figure, or the points of the solid
// See also: Polytope::constraints, Polytope::is_bounded
//
// CORA: contSet/@polytope/vertices_.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The vertices of the bounded polytope as rows; refuses a batch and an unbounded set.
Rows aux_vertices(const Polytope &P) {
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    if (like.ndim() != 2)
        CORA_FAIL(Error::not_supported, "vertices: it supports a single polytope, not a batch");
    if (!priv_polytope_is_bounded(P))
        CORA_FAIL(Error::special_error, "The vertices of an unbounded polytope are not defined.");
    return priv_polytope_vertices(P);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Polytope::vertices_impl() const {
    Rows points = aux_vertices(*this);
    if (points.empty()) return {};
    // counter-clockwise around the mean
    Row mean(2, 0.0);
    for (const Row &p : points)
        for (std::size_t i = 0; i < 2; ++i) mean[i] += p[i] / double(points.size());
    std::sort(points.begin(), points.end(), [&](const Row &a, const Row &b) {
        return std::atan2(a[1] - mean[1], a[0] - mean[0]) <
               std::atan2(b[1] - mean[1], b[0] - mean[0]);
    });
    Polygon polygon;
    for (const Row &p : points) polygon.push_back({p[0], p[1]});
    return {polygon};
}

std::vector<std::vector<Point3>> Polytope::vertices_3d_impl() const {
    const Rows points = aux_vertices(*this);
    if (points.empty()) return {};
    std::vector<Point3> solid;
    for (const Row &p : points) solid.push_back({p[0], p[1], p[2]});
    return {solid};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
