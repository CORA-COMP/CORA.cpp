// polygon_union - the union of polygons as closed loops
//
// Syntax:   loops = union_polygons(polygons);
// Inputs:   polygons that may overlap, touch or share edges
// Outputs:  outer boundaries counter-clockwise, holes clockwise
// See also: cora/global/plot/geometry.h, src/global/plot/geometry.cpp
//
// The union is its boundary: the parts of the edges that no other polygon covers. Each polygon is
// convex (a non-convex one is cut into triangles first), so the part of an edge inside another
// polygon is one interval, found by clipping the edge against the polygon's half-planes. The
// uncovered pieces keep their direction and chain into loops.

#include "cora/global/plot/geometry.h"
#include "global/vec.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Polygons ----------------------------------------------------------------------------------------

// A convex polygon with what makes tests against it fast: its bounds, a disc inside it and a disc
// around it, both centered at the mean of its corners, the polygon of about 16 of its corners,
// which lies inside it, and the polygon of the lines of about 16 of its edges, which holds it.
struct Convex {
    Polygon corners; // counter-clockwise
    Polygon core;
    std::vector<std::pair<Point, Point>> lines; // some edges, as points a -> b
    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
    Point center = {0, 0};
    double inner = 0, outer = 0;

    explicit Convex(Polygon p) : corners(std::move(p)) {
        // The bounds and the mean of the corners.
        for (const Point &v : corners) {
            x0 = std::min(x0, v[0]);
            x1 = std::max(x1, v[0]);
            y0 = std::min(y0, v[1]);
            y1 = std::max(y1, v[1]);
            center = {center[0] + v[0] / static_cast<double>(corners.size()),
                      center[1] + v[1] / static_cast<double>(corners.size())};
        }
        const std::size_t step = std::max<std::size_t>(1, corners.size() / 16);
        for (std::size_t i = 0; i < corners.size(); i += step) {
            core.push_back(corners[i]);
            lines.push_back({corners[i], corners[(i + 1) % corners.size()]});
        }
        // The distance from the center to the nearest edge line, and to the farthest corner.
        inner = 1e300;
        for (std::size_t i = 0; i < corners.size(); ++i) {
            const Point &a = corners[i], &b = corners[(i + 1) % corners.size()];
            inner = std::min(inner, cross(sub(b, a), sub(center, a)) / norm(sub(b, a)));
            outer = std::max(outer, norm(sub(corners[i], center)));
        }
    }

    bool meets(const Convex &o, double eps) const {
        return x0 <= o.x1 + eps && o.x0 <= x1 + eps && y0 <= o.y1 + eps && o.y0 <= y1 + eps;
    }
};

// Counter-clockwise, without repeated neighbors.
Polygon aux_ccw(const Polygon &polygon, double eps) {
    Polygon p;
    for (const Point &v : polygon)
        if (p.empty() || norm(sub(v, p.back())) > eps) p.push_back(v);
    while (p.size() > 1 && norm(sub(p.front(), p.back())) <= eps) p.pop_back();
    if (area(p) < 0) std::reverse(p.begin(), p.end());
    return p;
}

// Whether every corner turns left, or goes straight on.
bool aux_is_convex(const Polygon &p, double eps) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Point &a = p[i], &b = p[(i + 1) % p.size()], &c = p[(i + 2) % p.size()];
        if (cross(sub(b, a), sub(c, b)) < -eps * norm(sub(b, a)) * norm(sub(c, b))) return false;
    }
    return true;
}

// Whether the point is inside the triangle (counter-clockwise), on its edges included.
bool aux_in_triangle(const Point &p, const Point &a, const Point &b, const Point &c) {
    return cross(sub(b, a), sub(p, a)) >= 0 && cross(sub(c, b), sub(p, b)) >= 0 &&
           cross(sub(a, c), sub(p, c)) >= 0;
}

// The triangles of a simple polygon by cutting off ears; none if it is not simple.
std::vector<Polygon> aux_triangles(const Polygon &p, double eps) {
    std::vector<std::size_t> idx(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) idx[i] = i;
    std::vector<Polygon> out;
    while (idx.size() > 3) {
        bool cut = false;
        for (std::size_t k = 0; k < idx.size() && !cut; ++k) {
            const Point &a = p[idx[(k + idx.size() - 1) % idx.size()]], &b = p[idx[k]],
                        &c = p[idx[(k + 1) % idx.size()]];
            if (cross(sub(b, a), sub(c, b)) <= eps * eps) continue; // not a convex corner
            bool empty = true;
            for (std::size_t j : idx)
                if (&p[j] != &a && &p[j] != &b && &p[j] != &c && aux_in_triangle(p[j], a, b, c))
                    empty = false;
            if (!empty) continue;
            out.push_back({a, b, c});
            idx.erase(idx.begin() + static_cast<std::ptrdiff_t>(k));
            cut = true;
        }
        if (!cut) return {};
    }
    out.push_back({p[idx[0]], p[idx[1]], p[idx[2]]});
    return out;
}

// The convex polygons of the input, counter-clockwise; none if one cannot be cut into convex parts.
bool aux_convex_polygons(const std::vector<Polygon> &polygons, double eps,
                         std::vector<Convex> &out) {
    for (const Polygon &given : polygons) {
        const Polygon p = aux_ccw(given, eps);
        if (p.size() < 3 || std::abs(area(p)) <= eps * eps) continue;
        if (aux_is_convex(p, 1e-12)) {
            out.emplace_back(p);
            continue;
        }
        const std::vector<Polygon> parts = aux_triangles(p, eps);
        if (parts.empty()) return false;
        for (const Polygon &t : parts)
            if (area(t) > eps * eps) out.emplace_back(t);
    }
    return true;
}

// Edges -------------------------------------------------------------------------------------------

// Whether the point is inside the polygon by more than eps.
bool aux_deep_inside(const Point &p, const Polygon &poly, double eps) {
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const Point &a = poly[i], &b = poly[(i + 1) % poly.size()];
        if (cross(sub(b, a), sub(p, a)) <= eps * norm(sub(b, a))) return false;
    }
    return true;
}

// Where the edge from P along d (t in [0, 1]) lies in the convex polygon J, if anywhere: the
// interval [t0, t1], found by clipping at every edge of J. Along an edge of J with the same
// direction the edge counts as inside only for the polygon of the lower index, so that of two
// equal edges one is kept; along an edge of the other direction it counts as inside, so that
// edges between two polygons vanish.
bool aux_inside(const Point &P, const Point &d, const Convex &J, bool lower, double eps, double &t0,
                double &t1) {
    const Point Q = {P[0] + d[0], P[1] + d[1]};
    // An edge between two points deep inside the polygon (its disc, or its core) is all inside.
    const bool in_disc =
        norm(sub(P, J.center)) < J.inner - eps && norm(sub(Q, J.center)) < J.inner - eps;
    if (in_disc || (aux_deep_inside(P, J.core, eps) && aux_deep_inside(Q, J.core, eps))) {
        t0 = 0;
        t1 = 1;
        return true;
    }
    if (segment_distance(J.center, P, Q) > J.outer + eps) return false;
    for (const auto &[a, b] : J.lines) { // beyond one of the lines of the polygon: outside it
        const Point f = sub(b, a);
        const double len = norm(f);
        if (cross(f, sub(P, a)) < -eps * len && cross(f, sub(Q, a)) < -eps * len) return false;
    }
    t0 = 0;
    t1 = 1;
    for (std::size_t i = 0; i < J.corners.size(); ++i) {
        const Point &a = J.corners[i], &b = J.corners[(i + 1) % J.corners.size()];
        const Point f = sub(b, a);
        const double len = norm(f);
        const double g0 = cross(f, sub(P, a)) / len, g1 = cross(f, sub(Q, a)) / len;
        if (std::abs(g0) <= eps && std::abs(g1) <= eps) { // along the edge
            if (dot(f, d) > 0 && !lower) return false;
            continue;
        }
        if (g0 < -eps && g1 < -eps) return false;
        if (g1 == g0) continue;
        // The edge crosses the line at t: it is inside after t if the distance rises, else before.
        const double t = -g0 / (g1 - g0);
        if (g1 > g0)
            t0 = std::max(t0, t);
        else
            t1 = std::min(t1, t);
        if (t1 <= t0) return false;
    }
    return true;
}

struct Piece {
    Point a, b;
};

// The polygons whose bounds meet those of each polygon, by a sweep over the left bounds.
std::vector<std::vector<std::size_t>> aux_neighbors(const std::vector<Convex> &polys, double eps) {
    std::vector<std::size_t> order(polys.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t p, std::size_t q) { return polys[p].x0 < polys[q].x0; });
    std::vector<std::vector<std::size_t>> near(polys.size());
    for (std::size_t a = 0; a < order.size(); ++a)
        for (std::size_t b = a + 1;
             b < order.size() && polys[order[b]].x0 <= polys[order[a]].x1 + eps; ++b)
            if (polys[order[a]].meets(polys[order[b]], eps)) {
                near[order[a]].push_back(order[b]);
                near[order[b]].push_back(order[a]);
            }
    return near;
}

// The pieces of the edges of polygon i that no other polygon covers: the edge without the union of
// the intervals that the neighbors cover.
void aux_boundary_pieces(const std::vector<Convex> &polys, const std::vector<std::size_t> &near,
                         std::size_t i, double eps, std::vector<Piece> &out) {
    const Polygon &poly = polys[i].corners;
    for (std::size_t k = 0; k < poly.size(); ++k) {
        const Point &p = poly[k], &q = poly[(k + 1) % poly.size()];
        const Point d = sub(q, p);
        const double slack = eps / norm(d);
        std::vector<std::pair<double, double>> covered;
        for (std::size_t j : near) {
            double t0 = 0, t1 = 0;
            if (aux_inside(p, d, polys[j], j < i, eps, t0, t1)) covered.push_back({t0, t1});
        }
        std::sort(covered.begin(), covered.end());
        // Walk along the edge: what lies before the next covered interval is a piece.
        double at = 0;
        auto emit = [&](double from, double to) {
            if (to - from > slack)
                out.push_back({{p[0] + from * d[0], p[1] + from * d[1]},
                               {p[0] + to * d[0], p[1] + to * d[1]}});
        };
        for (const auto &[t0, t1] : covered) {
            if (t0 > at) emit(at, t0);
            at = std::max(at, t1);
        }
        emit(at, 1.0);
    }
}

// Loops -------------------------------------------------------------------------------------------

// Chains the directed pieces into closed loops; false if one does not close.
bool aux_chain(const std::vector<Piece> &pieces, double eps, std::vector<Polygon> &loops) {
    const double cell = 8 * eps;
    auto key = [&](const Point &p) {
        return std::make_pair(static_cast<long long>(std::floor(p[0] / cell)),
                              static_cast<long long>(std::floor(p[1] / cell)));
    };
    auto hash = [](const std::pair<long long, long long> &k) {
        return std::hash<long long>()(k.first * 1000003LL + k.second);
    };
    std::unordered_map<std::pair<long long, long long>, std::vector<std::size_t>, decltype(hash)>
        starts(16, hash);
    for (std::size_t i = 0; i < pieces.size(); ++i) starts[key(pieces[i].a)].push_back(i);
    std::vector<bool> used(pieces.size(), false);
    // The first unused piece that starts within 4 eps of p.
    auto next_from = [&](const Point &p) -> long {
        const auto [kx, ky] = key(p);
        for (long long dx = -1; dx <= 1; ++dx)
            for (long long dy = -1; dy <= 1; ++dy) {
                const auto found = starts.find({kx + dx, ky + dy});
                if (found == starts.end()) continue;
                for (std::size_t i : found->second)
                    if (!used[i] && norm(sub(pieces[i].a, p)) <= 4 * eps)
                        return static_cast<long>(i);
            }
        return -1;
    };
    // Start a loop at every unused piece and follow it until it comes back.
    for (std::size_t first = 0; first < pieces.size(); ++first) {
        if (used[first]) continue;
        Polygon loop = {pieces[first].a};
        used[first] = true;
        Point end = pieces[first].b;
        while (norm(sub(end, loop[0])) > 4 * eps) {
            const long i = next_from(end);
            if (i < 0) return false;
            used[static_cast<std::size_t>(i)] = true;
            loop.push_back(pieces[static_cast<std::size_t>(i)].a);
            end = pieces[static_cast<std::size_t>(i)].b;
        }
        loops.push_back(std::move(loop));
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Union -------------------------------------------------------------------------------------------

std::vector<Polygon> union_polygons(const std::vector<Polygon> &given, double tol) {
    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
    for (const Polygon &p : given)
        for (const Point &v : p) {
            x0 = std::min(x0, v[0]);
            x1 = std::max(x1, v[0]);
            y0 = std::min(y0, v[1]);
            y1 = std::max(y1, v[1]);
        }
    const double eps = 1e-9 * std::max({x1 - x0, y1 - y0, 1e-300});
    // Corners that the picture cannot show are dropped first: fewer edges to test.
    std::vector<Polygon> polygons;
    for (const Polygon &p : given)
        polygons.push_back(thin_polygon(p, {tol * (x1 - x0), tol * (y1 - y0)}));
    // The polygons with an area, convex and counter-clockwise; their neighbors.
    std::vector<Convex> polys;
    const bool simple = aux_convex_polygons(polygons, eps, polys);
    if (!simple || polys.size() < 2) {
        std::vector<Polygon> same;
        for (const Polygon &p : polygons)
            if (const Polygon q = aux_ccw(p, eps); q.size() >= 3) same.push_back(q);
        return same;
    }
    const std::vector<std::vector<std::size_t>> near = aux_neighbors(polys, eps);
    // The boundary is what is left of the edges; it is chained into loops, or the polygons are
    // returned if that fails.
    std::vector<Piece> pieces;
    for (std::size_t i = 0; i < polys.size(); ++i)
        aux_boundary_pieces(polys, near[i], i, eps, pieces);
    std::vector<Polygon> loops;
    if (!aux_chain(pieces, eps, loops)) {
        std::vector<Polygon> same;
        for (const Convex &c : polys) same.push_back(c.corners);
        return same;
    }
    std::vector<Polygon> out;
    for (const Polygon &loop : loops) {
        Polygon thin = thin_polygon(loop, {eps, eps});
        if (thin.size() >= 3) out.push_back(std::move(thin));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
