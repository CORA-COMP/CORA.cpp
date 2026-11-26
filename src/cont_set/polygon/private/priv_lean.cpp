// priv_lean - the pieces on the host and CORALean's tests of them, rounded as CORALean rounds
//
// CORALean's float definitions at nearest: sumDown and dotUp nest to the right and end with + 0
// (which turns a -0 into +0), maximumOver starts at the last element, a strict test is the
// machine's <. Compiled without fused multiply-add (CMakeLists.txt), as every product rounds.
//
// Syntax:   double o = priv_orient(p, q, r);   bool b = priv_piece_holds(v, x);
// See also: priv.h, contains.cpp, is_intersecting.cpp, isequal.cpp
//
// CORA.cpp only: the helpers of polygon's operations, after CORALean's Polygon float definitions
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"

#include <cstddef>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// Crossing ----------------------------------------------------------------------------------------

Piece priv_host(const Polytope &P) {
    const std::vector<double> data = P.V().data();
    Piece v(data.size() / 2);
    for (std::size_t l = 0; l < v.size(); ++l) v[l] = {data[2 * l], data[2 * l + 1]};
    return v;
}

std::vector<Piece> priv_hosts(const Polygon &pgon) {
    std::vector<Piece> out;
    out.reserve(pgon.pieces().size());
    for (const Polytope &P : pgon.pieces()) out.push_back(priv_host(P));
    return out;
}

Polytope priv_piece(const Piece &v, const Tensor &like) {
    std::vector<double> data;
    data.reserve(2 * v.size());
    // the vertices as rows (p, 2)
    for (const Point &x : v) data.insert(data.end(), {x[0], x[1]});
    return Polytope(Tensor::like(like, data, {int64_t(v.size()), 2}));
}

Polygon priv_polygon_of(const std::vector<Piece> &pieces, const Tensor &like) {
    std::vector<Polytope> out;
    out.reserve(pieces.size());
    for (const Piece &v : pieces) out.push_back(priv_piece(v, like));
    return priv_polygon(std::move(out), like);
}

// CORALean's float definitions ----------------------------------------------------------------

double priv_maximum(double a, double b) { return a <= b ? b : (b < a ? a : a + b); }

double priv_minimum(double a, double b) { return a <= b ? a : (b < a ? b : a + b); }

double priv_orient(Point p, Point q, Point r) {
    // the six products of orientBounds, summed from the last: t0 + (t1 + (... + (t5 + 0)))
    const double t[6] = {q[0] * r[1],    p[0] * q[1],    r[0] * p[1],
                         -(q[0] * p[1]), -(p[0] * r[1]), -(r[0] * q[1])};
    double sum = 0.0;
    for (int i = 5; i >= 0; --i) sum = t[i] + sum;
    return sum;
}

bool priv_distinct(Point a, Point b, Point c) {
    // CORALean's distinctPoints: two points agree when each coordinate is <= the other's
    const auto same = [](Point p, Point q) {
        return p[0] <= q[0] && q[0] <= p[0] && p[1] <= q[1] && q[1] <= p[1];
    };
    return !same(a, b) && !same(b, c) && !same(c, a);
}

bool priv_piece_holds(const Piece &p, Point x) {
    // inTriangle at every ordered triple: a left turn a, b, c, and x inside each edge
    for (const Point &a : p)
        for (const Point &b : p)
            for (const Point &c : p)
                if (priv_distinct(a, b, c) && 0.0 < priv_orient(a, b, c) &&
                    0.0 <= priv_orient(b, c, x) && 0.0 <= priv_orient(c, a, x) &&
                    0.0 <= priv_orient(a, b, x))
                    return true;
    return false;
}

bool priv_piece_subset(const Piece &p, const Piece &q) {
    for (const Point &x : p)
        if (!priv_piece_holds(q, x)) return false;
    return true;
}

double priv_bound_dot(const Piece &p, Point a) {
    if (p.empty()) return 0.0 + 0.0;
    // maximumOver of the dot products, from the last vertex
    double best = a[0] * p.back()[0] + (a[1] * p.back()[1] + 0.0);
    for (std::size_t l = p.size() - 1; l-- > 0;)
        best = priv_maximum(a[0] * p[l][0] + (a[1] * p[l][1] + 0.0), best);
    // the bound of the empty error box is +0
    return best + 0.0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
