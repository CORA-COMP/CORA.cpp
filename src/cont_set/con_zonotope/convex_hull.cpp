// convex_hull - the convex hull of a constrained zonotope and other sets, as CORA's convHull_
//
// The hull has one more generator for the weight of the sets and one slack factor per bound. A
// list of constrained zonotopes gives the hull of them all and this set (CORA's list form leaves
// the first operand out, docs/cora-differences.md).
//
// Syntax:   cZres = cz.convex_hull(S);   cZres = cz.convex_hull(S, "exact:Kochdumper");
//           cZres = cz.convex_hull({cz2, cz3});
// Inputs:   S - a constrained zonotope, a zonotope, a box or a point (n, 1) of the same dimension
//           method - "exact" and "exact:Raghuraman" (default) or "exact:Kochdumper"
// Outputs:  cZres - exact: the convex hull as a constrained zonotope
// See also: enclose, Zonotope::convex_hull
//
// CORA: contSet/@conZonotope/convHull_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Hulls of two sets -------------------------------------------------------------------------------

/// The hull by Raghuraman and Koeln: the weights of the two sets are one factor, and slack factors
/// turn the bounds of the scaled factors into equalities.
ConZonotope aux_raghuraman(const ConZonotope &cz1, const ConZonotope &cz2) {
    const Tensor &c1 = cz1.c(), &c2 = cz2.c(), &G1 = cz1.generators(), &G2 = cz2.generators();
    const int64_t n = cz1.dim(), m1 = G1.shape().back(), m2 = G2.shape().back();
    const int64_t p1 = cz1.A().shape()[cz1.A().ndim() - 2],
                  p2 = cz2.A().shape()[cz2.A().ndim() - 2];
    const int64_t s = 2 * (m1 + m2);
    const Tensor c = (c1 + c2) * 0.5;
    const Tensor G = Tensor::concat({G1, G2, (c1 - c2) * 0.5, priv_zeros(G1, n, s)}, -1);
    // the rows of the weight: the scaled factors stay within the box of their set
    const Tensor A31 = Tensor::concat(
        {priv_identity(G1, m1), -priv_identity(G1, m1), priv_zeros(G1, 2 * m2, m1)}, -2);
    const Tensor A32 = Tensor::concat(
        {priv_zeros(G1, 2 * m1, m2), priv_identity(G1, m2), -priv_identity(G1, m2)}, -2);
    const Tensor A30 =
        Tensor::concat({priv_full(G1, 2 * m1, 1, -0.5), priv_full(G1, 2 * m2, 1, 0.5)}, -2);
    const Tensor row1 = Tensor::concat(
        {cz1.A(), priv_zeros(G1, p1, m2), cz1.b() * -0.5, priv_zeros(G1, p1, s)}, -1);
    const Tensor row2 =
        Tensor::concat({priv_zeros(G1, p2, m1), cz2.A(), cz2.b() * 0.5, priv_zeros(G1, p2, s)}, -1);
    const Tensor row3 = Tensor::concat({A31, A32, A30, priv_identity(G1, s)}, -1);
    return {c, G, Tensor::concat({row1, row2, row3}, -2),
            Tensor::concat({cz1.b() * 0.5, cz2.b() * 0.5, priv_full(G1, s, 1, -0.5)}, -2)};
}

/// The hull by Kochdumper: three copies of the weight and the factors of each set.
ConZonotope aux_kochdumper(const ConZonotope &cz1, const ConZonotope &cz2) {
    const Tensor &c1 = cz1.c(), &c2 = cz2.c(), &G1 = cz1.generators(), &G2 = cz2.generators();
    const int64_t n = cz1.dim(), m1 = G1.shape().back(), m2 = G2.shape().back();
    // a set without constraints gets the row 0 = 0
    const bool none1 = cz1.A().shape()[cz1.A().ndim() - 2] == 0;
    const bool none2 = cz2.A().shape()[cz2.A().ndim() - 2] == 0;
    const Tensor A1 = none1 ? priv_zeros(G1, 1, m1) : cz1.A(),
                 b1 = none1 ? priv_zeros(G1, 1, 1) : cz1.b();
    const Tensor A2 = none2 ? priv_zeros(G1, 1, m2) : cz2.A(),
                 b2 = none2 ? priv_zeros(G1, 1, 1) : cz2.b();
    const int64_t p = A1.shape()[A1.ndim() - 2] + A2.shape()[A2.ndim() - 2];
    const Tensor G =
        Tensor::concat({G1, G2, (c1 - c2) * 0.5, priv_zeros(G1, n, 3 * (m1 + m2))}, -1);
    // the constraints of both sets, with the weight in the last column
    const Tensor row1 =
        Tensor::concat({priv_block_diag(A1, A2), Tensor::concat({b1 * -0.5, b2 * 0.5}, -2),
                        priv_zeros(G1, p, 3 * (m1 + m2))},
                       -1);
    // the factors of each set against the weight: -2 I on the slack factors and its own copy
    const Tensor slack1 = Tensor::concat(
        {priv_identity(G1, 2 * m1) * -2.0,
         Tensor::concat({priv_identity(G1, m1) * -2.0, priv_identity(G1, m1) * -2.0}, -2)},
        -1);
    const Tensor slack2 = Tensor::concat(
        {priv_identity(G1, 2 * m2) * -2.0,
         Tensor::concat({priv_identity(G1, m2) * -2.0, priv_identity(G1, m2) * -2.0}, -2)},
        -1);
    const Tensor row2 = Tensor::concat(
        {Tensor::concat({priv_identity(G1, m1) * -2.0, priv_identity(G1, m1) * 2.0}, -2),
         priv_zeros(G1, 2 * m1, m2), priv_full(G1, 2 * m1, 1, 1.0), slack1,
         priv_zeros(G1, 2 * m1, 3 * m2)},
        -1);
    const Tensor row3 = Tensor::concat(
        {priv_zeros(G1, 2 * m2, m1),
         Tensor::concat({priv_identity(G1, m2) * 2.0, priv_identity(G1, m2) * -2.0}, -2),
         priv_full(G1, 2 * m2, 1, -1.0), priv_zeros(G1, 2 * m2, 3 * m1), slack2},
        -1);
    const Tensor A = Tensor::concat({row1, row2, row3}, -2);
    const Tensor b = Tensor::concat({b1 * 0.5, b2 * 0.5, priv_full(G1, 2 * (m1 + m2), 1, 3.0)}, -2);
    // the rows that are zero are dropped
    const std::vector<int64_t> rows = priv_nonzero_rows(A, b);
    return {(c1 + c2) * 0.5, G, A.take(rows, -2), b.take(rows, -2)};
}

// Hull of many sets -------------------------------------------------------------------------------

/// The hull of several sets: the weights of all sets add to one.
ConZonotope aux_many(const std::vector<ConZonotope> &sets) {
    const Tensor &like = sets.front().generators();
    const int64_t n = sets.front().dim(), N = int64_t(sets.size());
    Tensor c = sets.front().c().zeros_like(), A = priv_zeros(like, 0, 0),
           b = priv_zeros(like, 0, 1);
    std::vector<Tensor> Gs;
    Tensor weights = priv_zeros(like, 1, 0);
    for (const ConZonotope &S : sets) {
        const int64_t m = S.generators().shape().back(), p = S.A().shape()[S.A().ndim() - 2];
        const Tensor o = priv_full(like, m, 1, 1.0), I = priv_identity(like, m),
                     O = priv_zeros(like, m, m);
        const Tensor O_ = priv_zeros(like, p, m);
        // the rows of the set (if it has constraints), then the two bounds of its factors
        const Tensor bounds = Tensor::concat({o, I * -2.0, I * -2.0, O, I * -2.0}, -1);
        const Tensor bounds2 = Tensor::concat({o, I * 2.0, O, I * -2.0, I * -2.0}, -1);
        const Tensor own = Tensor::concat({S.b() * -0.5, S.A(), O_, O_, O_}, -1);
        A = priv_block_diag(A, Tensor::concat({own, bounds, bounds2}, -2));
        b = Tensor::concat({b, S.b() * 0.5, priv_full(like, 2 * m, 1, 3.0)}, -2);
        weights =
            Tensor::concat({weights, priv_full(like, 1, 1, 0.5), priv_zeros(like, 1, 4 * m)}, -1);
        Gs.push_back(Tensor::concat({S.c() * 0.5, S.generators(), priv_zeros(like, n, 3 * m)}, -1));
        c = c + S.c() * 0.5;
    }
    return {c, Tensor::concat(Gs, -1), Tensor::concat({A, weights}, -2),
            Tensor::concat({b, priv_full(like, 1, 1, 1.0 - 0.5 * double(N))}, -2)};
}

/// Why the hull is refused, if it is: the sets must have one dimension, and the method must be one.
// Refusals and dispatch ---------------------------------------------------------------------------

detail::Refusal aux_refusal(const ConZonotope &cz, int64_t other_dim, const std::string &method) {
    if (other_dim != cz.dim())
        return std::pair{Error::dimension_mismatch, "convex_hull: the sets differ in dimension, " +
                                                        std::to_string(cz.dim()) + " and " +
                                                        std::to_string(other_dim)};
    if (method == "exact" || method == "exact:Raghuraman" || method == "exact:Kochdumper")
        return std::nullopt;
    return std::pair{Error::wrong_value,
                     "convex_hull: the method '" + method +
                         "' is unknown; use 'exact', 'exact:Raghuraman' or 'exact:Kochdumper'"};
}

/// The hull of two sets by the method.
ConZonotope aux_hull(const ConZonotope &cz1, const ConZonotope &cz2, const std::string &method) {
    if (method == "exact:Kochdumper") return aux_kochdumper(cz1, cz2);
    return aux_raghuraman(cz1, cz2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::convex_hull(const ConZonotope &S, const std::string &method) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_hull(*this, S, method);
}

ConZonotope ConZonotope::convex_hull(const Zonotope &Z, const std::string &method) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, Z.dim(), method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_hull(*this, ConZonotope(Z), method);
}

ConZonotope ConZonotope::convex_hull(const Interval &I, const std::string &method) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, I.dim(), method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_hull(*this, ConZonotope(I), method);
}

ConZonotope ConZonotope::convex_hull(const Tensor &p, const std::string &method) const {
    detail::Entry entry;
    // the hull of the set and a point
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        // a point is a set without generators
        return "convex_hull: the point must be a column (" + std::to_string(dim()) +
               ", 1), got the shape " + shape_text(p.shape());
    });
    const detail::Refusal refusal = aux_refusal(*this, dim(), method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_hull(*this, ConZonotope(p, priv_zeros(p, dim(), 0)), method);
}

ConZonotope ConZonotope::convex_hull(const std::vector<ConZonotope> &others) const {
    detail::Entry entry;
    for (const ConZonotope &S : others) {
        const detail::Refusal refusal = aux_refusal(*this, S.dim(), "exact");
        CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    }
    std::vector<ConZonotope> sets{*this};
    sets.insert(sets.end(), others.begin(), others.end());
    return aux_many(sets);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
