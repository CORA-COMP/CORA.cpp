// priv_clip - the exact geometry of convex pieces: hulls, clipping, differences, areas
//
// CORA.cpp's own, for the operations CORALean answers with boxes (intersect, set_difference,
// volume, center, triangulation): a convex piece is clipped by halfplanes (Sutherland-Hodgman),
// so the results are exact up to the rounding of the crossing points. A part with fewer than three
// vertices has no area and is dropped, as polyshape drops it.
//
// Syntax:   Piece h = priv_hull(v);   Piece c = priv_clip(h, a, b);   double A = priv_area(h);
// See also: priv.h, intersect.cpp, set_difference.cpp, volume.cpp
//
// CORA.cpp only: the clipping of convex polygons that stands for MATLAB's polyshape
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"

#include <algorithm>
#include <cstddef>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// (b - a) x (c - a), positive for a left turn.
double aux_cross(Point a, Point b, Point c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/// The outward normal (d1, -d0) of the edge i of a counter-clockwise polygon and its offset.
std::pair<Point, double> aux_edge(const Piece &q, std::size_t i) {
    const Point a = q[i], b = q[(i + 1) % q.size()];
    const Point n = {b[1] - a[1], a[0] - b[0]};
    return {n, n[0] * a[0] + n[1] * a[1]};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Hulls and clipping ----------------------------------------------------------------------------

Piece priv_hull(const Piece &p) {
    Piece pts = p;
    std::sort(pts.begin(), pts.end(),
              [](Point a, Point b) { return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]); });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    if (pts.size() < 3) return pts;
    // Andrew's monotone chain, without collinear points: the lower chain, then the upper one
    Piece hull(2 * pts.size());
    std::size_t k = 0;
    for (const Point &x : pts) {
        while (k >= 2 && aux_cross(hull[k - 2], hull[k - 1], x) <= 0) --k;
        hull[k++] = x;
    }
    for (std::size_t i = pts.size() - 1, lower = k + 1; i-- > 0;) {
        while (k >= lower && aux_cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
        hull[k++] = pts[i];
    }
    hull.resize(k - 1);
    return hull;
}

Piece priv_clip(const Piece &c, Point a, double b) {
    Piece out;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const Point P = c[i], Q = c[(i + 1) % c.size()];
        const double s_p = a[0] * P[0] + a[1] * P[1] - b, s_q = a[0] * Q[0] + a[1] * Q[1] - b;
        if (s_p <= 0) out.push_back(P);
        // the edge crosses the line: the crossing point
        if ((s_p < 0 && s_q > 0) || (s_p > 0 && s_q < 0)) {
            const double t = s_p / (s_p - s_q);
            out.push_back({P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])});
        }
    }
    out.erase(std::unique(out.begin(), out.end()), out.end());
    while (out.size() > 1 && out.front() == out.back()) out.pop_back();
    // no area left: nothing
    if (out.size() < 3 || priv_area(out) <= 0) return {};
    return out;
}

// Differences and areas --------------------------------------------------------------------------

std::vector<Piece> priv_minus_piece(const Piece &r, const Piece &q) {
    if (q.size() < 3) return {r};
    std::vector<Piece> parts;
    Piece rest = r;
    // outside edge i and inside the edges before it: disjoint parts
    for (std::size_t i = 0; i < q.size() && !rest.empty(); ++i) {
        const auto [n, offset] = aux_edge(q, i);
        Piece outside = priv_clip(rest, {-n[0], -n[1]}, -offset);
        if (!outside.empty()) parts.push_back(std::move(outside));
        rest = priv_clip(rest, n, offset);
    }
    return parts;
}

std::vector<Piece> priv_disjoint(const std::vector<Piece> &pieces) {
    std::vector<Piece> hulls, parts;
    for (const Piece &p : pieces) {
        const Piece h = priv_hull(p);
        if (h.size() < 3) continue;
        // the part of this piece that the earlier ones miss
        std::vector<Piece> mine = {h};
        for (const Piece &q : hulls) {
            std::vector<Piece> next;
            for (const Piece &m : mine)
                for (Piece &d : priv_minus_piece(m, q)) next.push_back(std::move(d));
            mine = std::move(next);
        }
        parts.insert(parts.end(), mine.begin(), mine.end());
        hulls.push_back(h);
    }
    return parts;
}

double priv_area(const Piece &loop) {
    double twice = 0;
    for (std::size_t i = 0; i < loop.size(); ++i) {
        const Point a = loop[i], b = loop[(i + 1) % loop.size()];
        twice += a[0] * b[1] - b[0] * a[1];
    }
    return twice / 2;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
