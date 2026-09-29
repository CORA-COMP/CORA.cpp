// vertices - the corners of a two-dimensional zonotope, as CORA's zonotope.vertices
//
// Syntax:   V = Z.vertices();
// Inputs:   -
// Outputs:  V - one polygon per batch member, counter-clockwise, without repeated points
// See also: project, Interval::vertices

#include "contSet/zonotope/zonotope.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The generators, turned into the upper half-plane and sorted by angle, are the edges of the
// polygon: walk 2 g_i from the vertex c - sum(g) through all of them, and back.
Polygon aux_polygon(const double *c, const double *G, int64_t m) {
    std::vector<Point> g; // the generators (G is row-major (2, m))
    for (int64_t j = 0; j < m; ++j) {
        Point p = {G[j], G[m + j]};
        if (std::hypot(p[0], p[1]) <= 1e-14) continue;
        if (p[1] < 0 || (p[1] == 0 && p[0] < 0)) p = {-p[0], -p[1]};
        g.push_back(p);
    }
    if (g.empty()) return {{c[0], c[1]}};
    std::sort(g.begin(), g.end(), [](const Point &a, const Point &b) {
        return std::atan2(a[1], a[0]) < std::atan2(b[1], b[0]);
    });
    // Parallel generators point the same way here and act as their sum: no repeated corners.
    std::vector<Point> merged = {g[0]};
    for (std::size_t i = 1; i < g.size(); ++i) {
        Point &last = merged.back();
        const double cross = last[0] * g[i][1] - last[1] * g[i][0];
        if (std::abs(cross) <= 1e-12 * std::hypot(last[0], last[1]) * std::hypot(g[i][0], g[i][1]))
            last = {last[0] + g[i][0], last[1] + g[i][1]};
        else
            merged.push_back(g[i]);
    }
    g = merged;
    Point v = {c[0], c[1]};
    for (const Point &p : g) v = {v[0] - p[0], v[1] - p[1]};
    Polygon out = {v};
    for (const Point &p : g) out.push_back(v = {v[0] + 2 * p[0], v[1] + 2 * p[1]});
    for (std::size_t i = 0; i + 1 < g.size(); ++i)
        out.push_back(v = {v[0] - 2 * g[i][0], v[1] - 2 * g[i][1]});
    return out;
}

} // namespace

// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Zonotope::vertices() const {
    if (dim() != 2)
        throw std::invalid_argument("Zonotope::vertices: needs a two-dimensional zonotope; "
                                    "project it first, e.g. Z.project({0, 1})");
    const std::vector<double> cs = c.data(), Gs = G.data();
    const int64_t m = G.shape().back();
    const std::size_t batch = cs.size() / 2;
    std::vector<Polygon> out;
    for (std::size_t b = 0; b < batch; ++b)
        out.push_back(aux_polygon(&cs[2 * b], &Gs[static_cast<std::size_t>(2 * m) * b], m));
    return out;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
