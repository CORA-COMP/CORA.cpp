// geometry - the thinning of polygons and the faces of a convex hull
//
// Syntax:   p = thin_polygon(p, {0.01, 0.01});   faces = hull_faces(points);
// Inputs:   poly - a polygon, or points in space as the sets' vertices give them
// Outputs:  hull - a polygon with fewer vertices, the faces of the hull
// See also: cora/global/plot/geometry.h, cora/global/plot/figure.h
//
// CORA.cpp only: the polygons of a plot; MATLAB's plotting and polyshape do this in CORA.
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cora/global/plot/geometry.h"
#include "cora/global/tensor/tolerance.h"
#include "global/vec.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Douglas-Peucker ---------------------------------------------------------------------------------

// Marks in `keep` the points of idx (indices into pts, the first and last kept) that stay.
void aux_simplify(const std::vector<Point> &pts, const std::vector<std::size_t> &idx,
                  std::vector<bool> &keep) {
    std::vector<std::pair<std::size_t, std::size_t>> stack = {{0, idx.size() - 1}};
    keep[idx.front()] = keep[idx.back()] = true;
    while (!stack.empty()) {
        const auto [lo, hi] = stack.back();
        stack.pop_back();
        // The point farthest from the chord, if it is beyond the tolerance, splits the chain.
        double worst = 1.0; // the tolerance is 1 in the scaled coordinates
        std::size_t at = 0;
        for (std::size_t k = lo + 1; k < hi; ++k) {
            const double dist = segment_distance(pts[idx[k]], pts[idx[lo]], pts[idx[hi]]);
            if (dist > worst) {
                worst = dist;
                at = k;
            }
        }
        if (at == 0) continue;
        keep[idx[at]] = true;
        stack.emplace_back(lo, at);
        stack.emplace_back(at, hi);
    }
}

// The faces of a convex hull ---------------------------------------------------------------------

struct Triangle {
    std::size_t a, b, c;
    Point3 n; // outward unit normal
    double d; // n'x = d on the plane
    bool alive = true;
};

Triangle aux_triangle(const std::vector<Point3> &p, std::size_t a, std::size_t b, std::size_t c,
                      const Point3 &inside) {
    Point3 n = cross(sub(p[b], p[a]), sub(p[c], p[a]));
    n = scale(n, 1 / norm(n));
    if (dot(n, sub(inside, p[a])) > 0) { // the normal must point away from the inside
        std::swap(b, c);
        n = scale(n, -1);
    }
    return {a, b, c, n, dot(n, p[a])};
}

// The index of the point farthest from the measure `far`, and that distance.
template <class F>
std::pair<std::size_t, double> aux_farthest(std::size_t count, F far) {
    std::size_t best = 0;
    double value = -1;
    for (std::size_t i = 0; i < count; ++i)
        if (const double v = far(i); v > value) {
            value = v;
            best = i;
        }
    return {best, value};
}

// The corners of the planar point set (plane through o, normal n): the monotone chain.
std::vector<Point3> aux_planar_hull(const std::vector<Point3> &pts, const Point3 &o,
                                    const Point3 &n, const Point3 &e1, double eps) {
    const Point3 e2 = cross(n, e1);
    std::vector<std::pair<Point, Point3>> q;
    q.reserve(pts.size());
    for (const Point3 &p : pts) q.push_back({{dot(sub(p, o), e1), dot(sub(p, o), e2)}, p});
    std::sort(q.begin(), q.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
    std::vector<std::pair<Point, Point3>> h(2 * q.size());
    std::size_t k = 0;
    auto turn = [&](std::size_t i, std::size_t j, const Point &c) {
        const Point ab = sub(h[j].first, h[i].first);
        return cross(ab, sub(c, h[i].first)) - eps * norm(ab); // positive: a left turn
    };
    for (auto &i : q) { // lower chain
        while (k >= 2 && turn(k - 2, k - 1, i.first) <= 0) --k;
        h[k++] = i;
    }
    for (std::size_t i = q.size() - 1, t = k + 1; i > 0; --i) { // upper chain
        while (k >= t && turn(k - 2, k - 1, q[i - 1].first) <= 0) --k;
        h[k++] = q[i - 1];
    }
    std::vector<Point3> out;
    for (std::size_t i = 0; i + 1 < k; ++i) out.push_back(h[i].second);
    return out;
}

// The vertices of the triangles of one plane as a polygon, counter-clockwise from outside.
std::vector<Point3> aux_merge(const std::vector<Point3> &p, const std::vector<Triangle> &tris,
                              const std::vector<std::size_t> &group, double eps) {
    std::vector<Point3> pts;
    std::vector<std::size_t> ids;
    for (std::size_t t : group)
        for (std::size_t v : {tris[t].a, tris[t].b, tris[t].c})
            if (std::find(ids.begin(), ids.end(), v) == ids.end()) {
                ids.push_back(v);
                pts.push_back(p[v]);
            }
    const Triangle &first = tris[group[0]];
    const Point3 e1 = scale(sub(p[first.b], p[first.a]), 1 / norm(sub(p[first.b], p[first.a])));
    return aux_planar_hull(pts, p[first.a], first.n, e1, eps);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Thinning ----------------------------------------------------------------------------------------

Polygon thin_polygon(const Polygon &polygon, Point resolution, bool closed) {
    const std::size_t n = polygon.size();
    if (n < (closed ? 4u : 3u)) return polygon;
    std::vector<Point> pts; // in units of the resolution, so that the limit is 1
    for (const Point &p : polygon)
        pts.push_back({p[0] / std::max(resolution[0], numerics::kTiny),
                       p[1] / std::max(resolution[1], numerics::kTiny)});
    std::vector<bool> keep(n, false);
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    if (closed) {
        // Two far apart points split the loop into two chains.
        const auto far = aux_farthest(n, [&](std::size_t i) { return norm(sub(pts[i], pts[0])); });
        if (far.second <= 1.0) return polygon;
        const std::vector<std::size_t> first(idx.begin(), idx.begin() + far.first + 1);
        std::vector<std::size_t> second(idx.begin() + far.first, idx.end());
        second.push_back(0);
        aux_simplify(pts, first, keep);
        aux_simplify(pts, second, keep);
    } else {
        aux_simplify(pts, idx, keep);
    }
    Polygon out;
    for (std::size_t i = 0; i < n; ++i)
        if (keep[i]) out.push_back(polygon[i]);
    return closed && out.size() < 3 ? polygon : out;
}

// Hull --------------------------------------------------------------------------------------------

std::vector<std::vector<Point3>> hull_faces(const std::vector<Point3> &p) {
    if (p.empty()) return {};
    Point3 lo = p[0], hi = p[0];
    for (const Point3 &v : p)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], v[k]);
            hi[k] = std::max(hi[k], v[k]);
        }
    const double eps = numerics::kHullExtent *
                       std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], numerics::kTiny});
    // The first point, the farthest from it, from their line, from their plane: a tetrahedron.
    const std::size_t i0 = 0;
    const auto [i1, d1] =
        aux_farthest(p.size(), [&](std::size_t i) { return norm(sub(p[i], p[i0])); });
    if (d1 <= eps) return {{p[i0]}};
    const Point3 u = scale(sub(p[i1], p[i0]), 1 / d1);
    const auto [i2, d2] =
        aux_farthest(p.size(), [&](std::size_t i) { return norm(cross(u, sub(p[i], p[i0]))); });
    if (d2 <= eps) { // a segment: the ends along u
        const auto lowest =
            aux_farthest(p.size(), [&](std::size_t i) { return -dot(u, sub(p[i], p[i0])); });
        const auto highest =
            aux_farthest(p.size(), [&](std::size_t i) { return dot(u, sub(p[i], p[i0])); });
        return {{p[lowest.first], p[highest.first]}};
    }
    Point3 n = cross(u, sub(p[i2], p[i0]));
    n = scale(n, 1 / norm(n));
    const auto [i3, d3] =
        aux_farthest(p.size(), [&](std::size_t i) { return std::abs(dot(n, sub(p[i], p[i0]))); });
    if (d3 <= eps) return {aux_planar_hull(p, p[i0], n, u, eps)};
    const Point3 inside = scale(add(add(p[i0], p[i1]), add(p[i2], p[i3])), 0.25);
    std::vector<Triangle> tris = {
        aux_triangle(p, i0, i1, i2, inside), aux_triangle(p, i0, i1, i3, inside),
        aux_triangle(p, i0, i2, i3, inside), aux_triangle(p, i1, i2, i3, inside)};
    for (std::size_t k = 0; k < p.size(); ++k) {
        if (k == i0 || k == i1 || k == i2 || k == i3) continue;
        std::vector<std::size_t> visible;
        for (std::size_t t = 0; t < tris.size(); ++t)
            if (tris[t].alive && dot(tris[t].n, p[k]) - tris[t].d > eps) visible.push_back(t);
        // a point is visible if it lies above a face
        if (visible.empty()) continue;
        // The edges of the visible part whose neighbor across is not visible form the horizon.
        std::vector<std::pair<std::size_t, std::size_t>> edges;
        for (std::size_t t : visible)
            for (const auto &e :
                 {std::make_pair(tris[t].a, tris[t].b), std::make_pair(tris[t].b, tris[t].c),
                  std::make_pair(tris[t].c, tris[t].a)})
                edges.push_back(e);
        for (std::size_t t : visible) tris[t].alive = false;
        for (const auto &[a, b] : edges) {
            const bool twin =
                std::find(edges.begin(), edges.end(), std::make_pair(b, a)) != edges.end();
            if (!twin) tris.push_back(aux_triangle(p, a, b, k, inside));
        }
    }
    // The triangles of one plane are merged into one face.
    std::vector<std::vector<Point3>> faces;
    std::vector<bool> done(tris.size(), false);
    for (std::size_t t = 0; t < tris.size(); ++t) {
        if (!tris[t].alive || done[t]) continue;
        std::vector<std::size_t> group = {t};
        for (std::size_t s = t + 1; s < tris.size(); ++s)
            if (tris[s].alive && !done[s] && dot(tris[t].n, tris[s].n) > 1 - numerics::kCoplanar &&
                std::abs(tris[t].d - tris[s].d) <= 10 * eps)
                group.push_back(s);
        for (std::size_t s : group) done[s] = true;
        faces.push_back(aux_merge(p, tris, group, eps));
    }
    return faces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
