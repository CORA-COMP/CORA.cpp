// vertices_3d - the corners of a three-dimensional zonotope, as CORA's zonotope.vertices_
//
// Syntax:   V = Z.vertices_3d();
// Inputs:   -
// Outputs:  V - one list of corners per batch member (a flat zonotope gives its polygon, a
//               segment its two ends, a point itself)
// See also: vertices, Interval::vertices_3d
//
// CORA: contSet/@zonotope/vertices_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/vec.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The corners of the face of the zonotope whose outer normal is d: the generators not parallel to
// that face add with the sign of g'd; those in its plane form a planar zonotope, whose corners
// are found by turning a direction u around d and taking the sign of g'u between the angles at
// which one of them flips.
void aux_face_corners(const Point3 &c, const std::vector<Point3> &g, const Point3 &d, double tol,
                      std::vector<Point3> &out) {
    Point3 base = c;
    std::vector<Point3> flat;
    for (const Point3 &k : g) {
        if (std::abs(dot(k, d)) <= tol * norm(k))
            flat.push_back(k);
        else
            base = add(base, k, dot(k, d) > 0 ? 1 : -1);
    }
    const Point3 e1 = scale(flat[0], 1 / norm(flat[0]));
    const Point3 e2 = cross(d, e1);
    const double pi = std::acos(-1.0);
    std::vector<double> angles; // where a flat generator is perpendicular to u
    for (const Point3 &k : flat) {
        const double a = std::atan2(dot(k, e2), dot(k, e1)) + pi / 2;
        angles.push_back(std::fmod(a + 4 * pi, pi));
        angles.push_back(std::fmod(a + 4 * pi, pi) + pi);
    }
    std::sort(angles.begin(), angles.end());
    for (std::size_t i = 0; i < angles.size(); ++i) {
        const double next = i + 1 < angles.size() ? angles[i + 1] : angles[0] + 2 * pi;
        if (next - angles[i] < 1e-9) continue; // parallel generators flip together
        const double phi = (angles[i] + next) / 2;
        const Point3 w = add(scale(e1, std::cos(phi)), e2, std::sin(phi));
        Point3 v = base;
        for (const Point3 &k : flat) v = add(v, k, dot(k, w) > 0 ? 1 : -1);
        out.push_back(v);
    }
}

// The corners of c + sum [-1, 1] g_k, without repeats.
std::vector<Point3> aux_corners(const Point3 &c, const std::vector<Point3> &g) {
    std::vector<Point3> out;
    double extent = 0;
    for (const Point3 &k : g) extent = std::max(extent, norm(k));
    bool found = false;
    for (std::size_t i = 0; i < g.size(); ++i)
        for (std::size_t j = i + 1; j < g.size(); ++j) {
            Point3 d = cross(g[i], g[j]);
            const double len = norm(d);
            if (len <= 1e-10 * norm(g[i]) * norm(g[j])) continue; // parallel
            d = scale(d, 1 / len);
            found = true;
            aux_face_corners(c, g, d, 1e-10, out);
            aux_face_corners(c, g, scale(d, -1), 1e-10, out);
        }
    if (!found && !g.empty()) { // all parallel: a segment
        Point3 s = {0, 0, 0};
        for (const Point3 &k : g) s = add(s, k, dot(k, g[0]) > 0 ? 1 : -1);
        out = {add(c, s, -1), add(c, s)};
    }
    if (g.empty()) out = {c};
    // Different faces share corners: keep one of each.
    std::vector<Point3> unique;
    for (const Point3 &p : out) {
        const bool seen = std::any_of(unique.begin(), unique.end(), [&](const Point3 &q) {
            return norm(sub(p, q)) <= 1e-9 * (extent + 1e-300);
        });
        if (!seen) unique.push_back(p);
    }
    return unique;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Zonotope::vertices_3d_impl() const {
    const std::vector<double> cs = c_.data(), Gs = G_.data();
    const int64_t m = G_.shape().back();
    const std::size_t batch = cs.size() / 3;
    std::vector<std::vector<Point3>> out;
    for (std::size_t b = 0; b < batch; ++b) {
        const double *G = &Gs[static_cast<std::size_t>(3 * m) * b]; // row-major (3, m)
        std::vector<Point3> g;
        for (int64_t j = 0; j < m; ++j) {
            const Point3 p = {G[j], G[m + j], G[2 * m + j]};
            if (cora::norm(p) > 1e-14) g.push_back(p);
        }
        out.push_back(aux_corners({cs[3 * b], cs[3 * b + 1], cs[3 * b + 2]}, g));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
