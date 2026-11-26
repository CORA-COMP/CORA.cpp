// vertices - the outline of a two-dimensional capsule, as CORA's capsule.vertices_
//
// A point (up to 1e-8) gives its single vertex. Else the outline is two half circles of 500 points
// each around c + g and c - g, starting at the angle of g minus a quarter turn, and closed by its
// first point. One polygon per nonempty batch member.
//
// Syntax:   V = C.vertices();
// Inputs:   -
// Outputs:  V - the polygons
// See also: vertices_3d, Zonotope::vertices
//
// CORA: contSet/@capsule/vertices_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"
#include <cmath>
#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Outline> Capsule::vertices_impl() const {
    const std::vector<double> cs = cc().data(), gs = gc().data(), rs = r_.data();
    const std::vector<uint8_t> empty = empty_.flags(r_.shape());
    constexpr int kPoints = 500; // per half circle, as CORA
    std::vector<Outline> out;
    for (std::size_t b = 0; b < rs.size(); ++b) {
        if (empty[b]) continue;
        const Point c = {cs[2 * b], cs[2 * b + 1]}, g = {gs[2 * b], gs[2 * b + 1]};
        const double r = rs[b], length = std::hypot(g[0], g[1]);
        if (length <= numerics::kVerticesPoint && r <= numerics::kVerticesPoint) {
            out.push_back({c});
            continue;
        }
        // the half circle around `from` that starts at `start` and turns by half a circle
        const double angle0 = length > 0 ? std::atan2(g[1], g[0]) : 0.0;
        Outline outline;
        for (int half = 0; half < 2; ++half) {
            const double sign = half == 0 ? 1.0 : -1.0,
                         start = angle0 + (half == 0 ? -1 : 1) * std::numbers::pi / 2;
            for (int k = 0; k < kPoints; ++k) {
                const double a = start + std::numbers::pi * k / (kPoints - 1);
                outline.push_back(
                    {c[0] + sign * g[0] + r * std::cos(a), c[1] + sign * g[1] + r * std::sin(a)});
            }
        }
        outline.push_back(outline.front());
        out.push_back(outline);
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
