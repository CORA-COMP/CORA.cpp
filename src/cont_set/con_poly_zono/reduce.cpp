// reduce - a constrained polynomial zonotope of fewer generators, as CORA's reduce (girard)
//
// compact first; past order * n generators in the two blocks, the generators are ranked by
// Girard's ||g||_1 - ||g||_inf over [G; GI] (the larger first, ties in order), the (order - 1) n
// first stay (in the order of their block's ranking) and the rest becomes a box, n independent
// generators of the summed |g| (CORALean's reduceGirard); the constraint stays.
//
// Syntax:   S = cPZ.reduce(order);
// Inputs:   order - the order (>= 1)
// Outputs:  S - a set that holds this one
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/reduce.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"
#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Girard's metric of each row of the (r, n) host matrix g: sum |g| (from the last entry) minus
/// max |g| (nested from the last).
std::vector<double> aux_metric(const std::vector<double> &g, int64_t r, int64_t n) {
    std::vector<double> out(static_cast<std::size_t>(r));
    for (int64_t l = 0; l < r; ++l) {
        double s = 0, m = n == 0 ? 0.0 : std::abs(g[std::size_t(l * n + n - 1)]);
        for (int64_t i = n; i-- > 0;) s = std::abs(g[std::size_t(l * n + i)]) + s;
        for (int64_t i = n - 1; i-- > 0;) {
            const double a = std::abs(g[std::size_t(l * n + i)]);
            m = a <= m ? m : (m < a ? a : a + m);
        }
        out[std::size_t(l)] = s - m;
    }
    return out;
}

/// The indices 0..r-1 by decreasing metric, ties in order (CORALean's sortPerm, a merge sort).
std::vector<int64_t> aux_rank(const std::vector<double> &metric) {
    std::vector<int64_t> order(metric.size());
    std::iota(order.begin(), order.end(), int64_t(0));
    // ties by the index, so the order is the stable one
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        const double x = metric[std::size_t(a)], y = metric[std::size_t(b)];
        return !(x <= y) || (x <= y && y <= x && a < b);
    });
    return order;
}

/// The sums of |g_l| over the dropped rows l of g (r, n), each from the last row.
std::vector<double> aux_radius(const std::vector<double> &g, const std::vector<int64_t> &dropped,
                               int64_t n) {
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (std::size_t t = dropped.size(); t-- > 0;)
            out[std::size_t(i)] =
                std::abs(g[std::size_t(dropped[t] * n + i)]) + out[std::size_t(i)];
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value,
               [] { return std::string("reduce: the order must be at least 1"); });
    const ConPolyZono X = compact();
    const int64_t n = dim(), h = X.generators().shape()[0],
                  q = X.independent_generators().shape()[0];
    if (int64_t(order) * n >= h + q) return X;
    const int64_t K = (int64_t(order) - 1) * n;
    const std::vector<double> g = X.generators().data(), gi = X.independent_generators().data();
    // the joint ranking: how many dependent generators are among the K first
    std::vector<double> both = g;
    both.insert(both.end(), gi.begin(), gi.end());
    const std::vector<int64_t> joint = aux_rank(aux_metric(both, h + q, n));
    int64_t d = 0;
    for (int64_t t = 0; t < std::min(K, h + q); ++t) d += joint[std::size_t(t)] < h ? 1 : 0;
    // each block ranked by itself: its first d (K - d) stay, in that order
    const std::vector<int64_t> rd = aux_rank(aux_metric(g, h, n)),
                               ri = aux_rank(aux_metric(gi, q, n));
    const int64_t kd = std::min(d, h), ki = std::min(K - d, q);
    const std::vector<int64_t> keep_d(rd.begin(), rd.begin() + kd),
        drop_d(rd.begin() + kd, rd.end());
    const std::vector<int64_t> keep_i(ri.begin(), ri.begin() + ki),
        drop_i(ri.begin() + ki, ri.end());
    // the box of the dropped ones: the two row sums added
    const std::vector<double> r1 = aux_radius(g, drop_d, n), r2 = aux_radius(gi, drop_i, n);
    std::vector<double> r(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) r[std::size_t(i)] = r1[std::size_t(i)] + r2[std::size_t(i)];
    Exponents E;
    for (const int64_t l : keep_d) E.push_back(X.exponents()[std::size_t(l)]);
    const Tensor GI = Tensor::concat(
        {X.independent_generators().take(keep_i, 0), Tensor::like(X.c(), r, {n, 1}).diag()}, 0);
    return {PolyZonotope(X.c(), X.generators().take(keep_d, 0), GI, std::move(E), X.ids(),
                         detail::Verbatim{}),
            X.cons_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
