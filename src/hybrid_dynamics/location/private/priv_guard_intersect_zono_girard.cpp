// priv_guard_intersect_zono_girard - the guard intersection of Girard, as CORA's zonoGirard
//
// For each basis vector l, each set is mapped to the zonogon (n'x, l'x), and the range of l'x on
// the line n'x = gamma is found by a dichotomic search over the sorted generators (Algorithms 2
// and 3 of Girard and Le Guernic, HSCC 2008). The ranges of the sets are united per basis, the box
// is mapped back, and the inequalities of the guard cut it (a constrained zonotope, rescaled). A
// bundle bounds each range by all of its zonotopes. Several bases give a zonotope bundle.
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_zono_girard(R, guard, B);
// Inputs:   R - the sets that cross the guard
//           guard - a hyperplane n'x = gamma with inequalities
//           B - the bases (n, n)
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_zonoGirard.m
// CORA authors: Stefan Liu, Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The zonogon -------------------------------------------------------------------------------------

/// A point or a vector of the plane.
struct Vec2 {
    double x = 0, y = 0;
};

/// The generators of a zonogon, as vectors.
using Gens = std::vector<Vec2>;

Vec2 aux_twice_sum(const Gens &G) {
    Vec2 s;
    for (const Vec2 &g : G) {
        s.x += 2 * g.x;
        s.y += 2 * g.y;
    }
    return s;
}

/// CORA's aux_lineIntersect2D: the height where the segment p1 p2 meets the line x = gamma.
std::optional<double> aux_line_intersect(const Vec2 &p1, const Vec2 &p2, double gamma) {
    const double eps = machine_eps();
    if (std::abs(p1.x - gamma) < eps) return p1.y;
    if (std::abs(p2.x - gamma) < eps) return p2.y;
    if (p1.x == p2.x) return std::nullopt; // a vertical segment off the line
    if (p1.y == p2.y) {
        // a horizontal segment meets the line between its ends
        if (gamma > std::min(p1.x, p2.x) && gamma < std::max(p1.x, p2.x)) return p2.y;
        return std::nullopt;
    }
    const double a = (p1.y - p2.y) / (p1.x - p2.x), b = p1.y - a * p1.x, y = a * gamma + b;
    if (y > std::max(p1.y, p2.y) || y < std::min(p1.y, p2.y)) return std::nullopt;
    return y;
}

/// CORA's aux_sort_trig: the generators by their angle, ascending (dir 1) or descending.
Gens aux_sort_by_angle(const Gens &g, int dir) {
    std::vector<std::size_t> order(g.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::vector<double> theta(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) theta[i] = std::atan2(g[i].y, g[i].x);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) {
        return dir == 1 ? theta[i] < theta[j] : theta[i] > theta[j];
    });
    Gens out;
    out.reserve(g.size());
    for (std::size_t i : order) out.push_back(g[i]);
    return out;
}

// The dichotomic search ---------------------------------------------------------------------------

/// One open branch of the search: the start point, the generators and their doubled sum.
struct Branch {
    Vec2 P;
    Gens G;
    Vec2 s;
};

/// CORA's aux_dichotomicSearch: the height where the boundary path from P along G meets x = gamma,
/// trying the branches put aside in their order when a path misses.
std::optional<double> aux_dichotomic_search(Branch start, double gamma, int dir) {
    std::deque<Branch> queue;
    Branch cur = std::move(start);
    for (;;) {
        Vec2 P = cur.P, s = cur.s;
        Gens G = std::move(cur.G);
        const std::size_t max_counter = G.size();
        for (std::size_t counter = 0; G.size() > 1 && counter < max_counter; ++counter) {
            // split at the pivot angle of the sum (CORA's aux_split_pivot)
            const double pivot = dir * s.x / std::sqrt(s.x * s.x + s.y * s.y);
            Gens G1, G2;
            for (const Vec2 &g : G)
                (dir * g.x / std::sqrt(g.x * g.x + g.y * g.y) <= pivot ? G1 : G2).push_back(g);
            const Vec2 s1 = aux_twice_sum(G1);
            if (aux_line_intersect(P, {P.x + s1.x, P.y + s1.y}, gamma)) {
                G = std::move(G1);
                s = s1;
            } else {
                // the first half is put aside, the path goes on through the second
                queue.push_back({P, std::move(G1), s1});
                G = std::move(G2);
                s = {s.x - s1.x, s.y - s1.y};
                P = {P.x + s1.x, P.y + s1.y};
            }
        }
        const std::optional<double> m = aux_line_intersect(P, {P.x + s.x, P.y + s.y}, gamma);
        if (m || queue.empty()) return m;
        cur = std::move(queue.front());
        queue.pop_front();
    }
}

/// CORA's aux_bound_intersect_2D: the range [m, M] of y on the line x = gamma in the zonogon.
std::optional<std::pair<double, double>> aux_bound_2d(const Vec2 &c, Gens g, double gamma) {
    // CORA's compact_(Z, 'zeros', eps), then every generator upward
    g.erase(std::remove_if(g.begin(), g.end(),
                           [](const Vec2 &v) {
                               return std::sqrt(v.x * v.x + v.y * v.y) <= machine_eps() ||
                                      (v.x == 0 && v.y == 0);
                           }),
            g.end());
    // a segment along the line (a clock's time-point set on its hyperplane): its own range
    if (std::all_of(g.begin(), g.end(), [](const Vec2 &v) { return v.x == 0; })) {
        if (std::abs(c.x - gamma) > numerics::kGuardTimePoint) return std::nullopt;
        double r = 0;
        for (const Vec2 &v : g) r += std::abs(v.y);
        return std::pair{c.y - r, c.y + r};
    }
    Vec2 low = c, high = c;
    for (Vec2 &v : g) {
        if (v.y < 0 || (v.y == 0 && v.x < 0)) v = {-v.x, -v.y};
        low = {low.x - v.x, low.y - v.y};
        high = {high.x + v.x, high.y + v.y};
    }
    // the lower bound from the lowest vertex, the upper from the highest
    const int dir_low = low.x < gamma ? 1 : -1;
    const Gens G_low = aux_sort_by_angle(g, dir_low);
    const std::optional<double> m =
        aux_dichotomic_search({low, G_low, aux_twice_sum(G_low)}, gamma, -dir_low);
    const int dir_high = high.x > gamma ? 1 : -1;
    Gens G_high = aux_sort_by_angle(g, dir_high);
    for (Vec2 &v : G_high) v = {-v.x, -v.y};
    const std::optional<double> M =
        aux_dichotomic_search({high, G_high, aux_twice_sum(G_high)}, gamma, dir_high);
    if (!m || !M) return std::nullopt;
    if (std::abs(*m) < numerics::kGuardFlat && std::abs(*M) < numerics::kGuardFlat)
        return std::pair{0.0, 0.0};
    if (std::abs(*m - *M) < numerics::kGuardFlat) return std::pair{*m, *m};
    return std::pair{*m, *M};
}

// The box per basis and the cut by the guard ------------------------------------------------------

/// CORA's aux_enclosingInterval: the box of the intersection in the coordinates of the basis B;
/// none if it is empty.
std::optional<Interval> aux_box(const Tensor &normal, double gamma, const Tensor &B, const Set &S) {
    const int64_t n = B.shape()[0];
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> lb(static_cast<std::size_t>(n), -inf), ub(static_cast<std::size_t>(n), inf);
    const Tensor rows = Tensor::concat({normal.matrix_transpose(), B.matrix_transpose()}, -2);
    for (const Zonotope &Z : priv_zonotopes(S)) {
        // the zonogons (n'x, l'x) of all basis vectors l at once
        const std::vector<double> c = rows.matmul(Z.center()).data();
        const Tensor GG = rows.matmul(Z.generators());
        const int64_t m = GG.shape().back();
        const std::vector<double> G = GG.data();
        auto entry = [&](std::size_t r, std::size_t j) {
            const double v = G[r * std::size_t(m) + j];
            return std::abs(v) < machine_eps() ? 0.0 : v;
        };
        const double cx = std::abs(c[0]) < machine_eps() ? 0.0 : c[0];
        for (std::size_t i = 0; i < std::size_t(n); ++i) {
            Gens g(static_cast<std::size_t>(m));
            for (std::size_t j = 0; j < std::size_t(m); ++j) g[j] = {entry(0, j), entry(i + 1, j)};
            const double cy = std::abs(c[i + 1]) < machine_eps() ? 0.0 : c[i + 1];
            const auto range = aux_bound_2d({cx, cy}, std::move(g), gamma);
            if (!range) return std::nullopt;
            lb[i] = std::max(lb[i], range->first);
            ub[i] = std::min(ub[i], range->second);
        }
    }
    // a basis vector along the normal takes the value of the hyperplane (CORA's robustProjection)
    const std::vector<double> n_b = B.matrix_transpose().matmul(normal).data();
    const double nn = normal.square().sum().item();
    std::vector<std::size_t> along;
    for (std::size_t i = 0; i < n_b.size(); ++i)
        if (std::abs(n_b[i]) >= machine_eps()) along.push_back(i);
    if (along.size() == 1) lb[along[0]] = ub[along[0]] = gamma * n_b[along[0]] / nn;
    // small inversions are rounding; a real one is an empty intersection
    for (std::size_t i = 0; i < lb.size(); ++i)
        if (ub[i] < lb[i]) ub[i] += machine_eps();
    for (std::size_t i = 0; i < lb.size(); ++i)
        if (lb[i] > ub[i]) return std::nullopt;
    return Interval(Tensor::like(B, lb, {n, 1}), Tensor::like(B, ub, {n, 1}));
}

/// CORA's aux_tightenSet: the zonotope cut by the inequalities P of the guard; none if it misses.
std::optional<Zonotope> aux_tighten(const Zonotope &Z, const Polytope &P) {
    if (!P.is_intersecting(Z, "approx", numerics::kGuardMeet)) return std::nullopt;
    const ConZonotope cz = ConZonotope(Z).intersect(P);
    ConZonotope tight = cz;
    try {
        tight = cz.rescale("iter");
    } catch (const Error &) { // CORA falls back to the linear programs
        tight = cz.rescale("exact");
    }
    return Zonotope(tight.c(), tight.generators());
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_zono_girard(const std::vector<Set> &R,
                                                    const Polytope &guard,
                                                    const std::vector<Tensor> &B) {
    const Tensor normal = guard.Ae().slice(-2, 0, 1).matrix_transpose();
    const double gamma = guard.be().data()[0];
    const bool inequalities = guard.A().shape()[0] > 0;
    std::vector<Zonotope> parts;
    for (const Tensor &basis : B) {
        // the union of the boxes of the sets in this basis
        std::optional<Interval> I;
        for (const Set &S : R)
            if (const std::optional<Interval> box = aux_box(normal, gamma, basis, S))
                I = I ? I->unite(*box) : *box;
        if (!I) return std::nullopt; // empty in one basis: empty
        Zonotope Z = I->zonotope().linear_map(basis);
        if (inequalities) {
            const std::optional<Zonotope> cut = aux_tighten(Z, Polytope(guard.A(), guard.b()));
            if (!cut) continue; // CORA drops the basis
            Z = *cut;
        }
        parts.push_back(std::move(Z));
    }
    if (parts.empty()) return std::nullopt;
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
