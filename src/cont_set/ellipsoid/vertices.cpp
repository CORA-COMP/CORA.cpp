// vertices - the outline of a two-dimensional ellipsoid
//
// The points q + G (cos t, sin t) for 256 equal steps of t, counter-clockwise, in the factor G of
// the symmetric root of Q; a point gives its single vertex. CORA supports vertices of ellipsoids in
// one dimension only and draws a 2D ellipsoid by its own points.
//
// Syntax:   V = E.vertices();
// Inputs:   -
// Outputs:  V - the polygons, one per batch member
// See also: vertices_3d, interval
//
// CORA: contSet/@ellipsoid/vertices_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

#include <cmath>
#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Ellipsoid::vertices_impl() const {
    if (priv_is_empty(*this)) CORA_FAIL(Error::empty_set, "vertices: the ellipsoid is empty");
    // the symmetric root of Q has a positive determinant, so the outline runs counter-clockwise
    const Tensor G = generators().matmul(Q_.svd()[0].matrix_transpose());
    const std::vector<double> qs = q_.data(), Gs = G.data();
    constexpr int kPoints = 256;
    std::vector<Polygon> out;
    for (std::size_t b = 0; b < qs.size() / 2; ++b) {
        const double *g = &Gs[4 * b];
        Polygon outline;
        for (int k = 0; k < kPoints; ++k) {
            const double t = 2.0 * std::numbers::pi * k / kPoints, c = std::cos(t), s = std::sin(t);
            outline.push_back(
                {qs[2 * b] + g[0] * c + g[1] * s, qs[2 * b + 1] + g[2] * c + g[3] * s});
        }
        out.push_back(outline);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
