// convex_hull - the convex hull of a constrained zonotope and other sets, as CORA's convHull_
//
// The hull has one more generator for the weight of the sets and one slack factor per bound. A
// list of constrained zonotopes gives the hull of them all and this set (CORA's list form leaves
// the first operand out, docs/cora-differences.md).
//
// Syntax:   cZres = cz.convex_hull(S);   cZres = cz.convex_hull(S, "exact:Kochdumper");
//           cZres = cz.convex_hull({cz2, cz3});
// Inputs:   S - a constrained zonotope, a zonotope, a box or a point (n) of the same dimension
//           method - "exact" and "exact:Raghuraman" (default) or "exact:Kochdumper"
// Outputs:  cZres - exact: the convex hull as a constrained zonotope
// See also: enclose, Zonotope::convex_hull
//
// CORA: contSet/@conZonotope/convHull_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/blocks.h"
#include "global/format.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Hulls of two sets -------------------------------------------------------------------------------

/// The center and the generators of a set as CORA's columns, (n, 1) and (n, m): the hulls stack
/// the factors of both sets along the columns.
std::pair<Tensor, Tensor> aux_columns(const ConZonotope &cz) {
    return {cz.c().expand_dims(-1), cz.generators().matrix_transpose()};
}

/// The hull by Raghuraman and Koeln: the weights of the two sets are one factor, and slack factors
/// turn the bounds of the scaled factors into equalities.
ConZonotope aux_raghuraman(const ConZonotope &cz1, const ConZonotope &cz2) {
    const auto [c1, G1] = aux_columns(cz1);
    const auto [c2, G2] = aux_columns(cz2);
    const int64_t n = cz1.dim(), m1 = cz1.nr_generators(), m2 = cz2.nr_generators();
    const int64_t s = 2 * (m1 + m2);
    const Tensor c = (c1 + c2) * 0.5;
    const Tensor G = Tensor::concat({G1, G2, (c1 - c2) * 0.5, zeros_block(G1, n, s)}, -1);
    // the rows of the weight: the scaled factors stay within the box of their set
    const Tensor A31 = Tensor::concat(
        {identity_block(G1, m1), -identity_block(G1, m1), zeros_block(G1, 2 * m2, m1)}, -2);
    const Tensor A32 = Tensor::concat(
        {zeros_block(G1, 2 * m1, m2), identity_block(G1, m2), -identity_block(G1, m2)}, -2);
    const Tensor A30 =
        Tensor::concat({full_block(G1, 2 * m1, 1, -0.5), full_block(G1, 2 * m2, 1, 0.5)}, -2);
    const Polytope weights =
        Polytope::equalities(Tensor::concat({A31, A32, A30, identity_block(G1, s)}, -1),
                             full_block(G1, s, 1, -0.5).squeeze(-1));
    // the rows of each set at the weight, free in the slack factors
    const Polytope F1 = cz1.factor_polytope().half_scaled_left(m2).pad_left(s);
    const Polytope F2 = cz2.factor_polytope().half_scaled_right(m1).pad_left(s);
    return {c.squeeze(-1), G.matrix_transpose(), F1.stack_rows(F2).stack_rows(weights)};
}

/// The hull by Kochdumper: three copies of the weight and the factors of each set.
ConZonotope aux_kochdumper(const ConZonotope &cz1, const ConZonotope &cz2) {
    const auto [c1, G1] = aux_columns(cz1);
    const auto [c2, G2] = aux_columns(cz2);
    const int64_t n = cz1.dim(), m1 = cz1.nr_generators(), m2 = cz2.nr_generators();
    const int64_t w = 3 * (m1 + m2);
    const Tensor G = Tensor::concat({G1, G2, (c1 - c2) * 0.5, zeros_block(G1, n, w)}, -1);
    // the rows of both sets, the weight last (CORA's 0 = 0 of a set without rows goes below)
    const Polytope own = cz1.factor_polytope()
                             .half_scaled_left(m2)
                             .stack_rows(cz2.factor_polytope().half_scaled_right(m1))
                             .pad_left(w);
    // the factors of each set against the weight: -2 I on the slack factors and its own copy
    const Tensor slack1 = Tensor::concat(
        {identity_block(G1, 2 * m1) * -2.0,
         Tensor::concat({identity_block(G1, m1) * -2.0, identity_block(G1, m1) * -2.0}, -2)},
        -1);
    // the same for the second set
    const Tensor slack2 = Tensor::concat(
        {identity_block(G1, 2 * m2) * -2.0,
         Tensor::concat({identity_block(G1, m2) * -2.0, identity_block(G1, m2) * -2.0}, -2)},
        -1);
    const Tensor row2 = Tensor::concat(
        {Tensor::concat({identity_block(G1, m1) * -2.0, identity_block(G1, m1) * 2.0}, -2),
         zeros_block(G1, 2 * m1, m2), full_block(G1, 2 * m1, 1, 1.0), slack1,
         zeros_block(G1, 2 * m1, 3 * m2)},
        -1);
    const Tensor row3 = Tensor::concat(
        {zeros_block(G1, 2 * m2, m1),
         Tensor::concat({identity_block(G1, m2) * 2.0, identity_block(G1, m2) * -2.0}, -2),
         full_block(G1, 2 * m2, 1, -1.0), zeros_block(G1, 2 * m2, 3 * m1), slack2},
        -1);
    const Polytope F = own.stack_rows(Polytope::equalities(
        Tensor::concat({row2, row3}, -2), full_block(G1, 2 * (m1 + m2), 1, 3.0).squeeze(-1)));
    // the rows that are zero are dropped
    return {((c1 + c2) * 0.5).squeeze(-1), G.matrix_transpose(),
            F.select_rows({}, priv_nonzero_rows(F.Ae(), F.be().expand_dims(-1)))};
}

// Hull of many sets -------------------------------------------------------------------------------

/// The hull of several sets: the weights of all sets add to one.
ConZonotope aux_many(const std::vector<ConZonotope> &sets) {
    const Tensor &like = sets.front().generators().matrix_transpose();
    const int64_t n = sets.front().dim(), N = int64_t(sets.size());
    Tensor c = sets.front().c().expand_dims(-1).zeros_like();
    Polytope F = Polytope::equalities(zeros_block(like, 0, 0), zeros_block(like, 0, 1).squeeze(-1));
    std::vector<Tensor> Gs;
    Tensor weights = zeros_block(like, 1, 0);
    for (const ConZonotope &S : sets) {
        const int64_t m = S.nr_generators();
        const Tensor o = full_block(like, m, 1, 1.0), I = identity_block(like, m),
                     O = zeros_block(like, m, m);
        // the rows of the set (if it has constraints) at its weight, the weight first
        std::vector<int64_t> weight_first{m};
        for (int64_t j = 0; j < m; ++j) weight_first.push_back(j);
        const Polytope own =
            S.factor_polytope().half_scaled_left(0).select_cols(weight_first).pad_left(3 * m);
        // then the two bounds of its factors
        const Tensor bounds = Tensor::concat({o, I * -2.0, I * -2.0, O, I * -2.0}, -1);
        const Tensor bounds2 = Tensor::concat({o, I * 2.0, O, I * -2.0, I * -2.0}, -1);
        const Polytope block = own.stack_rows(Polytope::equalities(
            Tensor::concat({bounds, bounds2}, -2), full_block(like, 2 * m, 1, 3.0).squeeze(-1)));
        // on the block diagonal of the rows so far
        F = F.pad_left(4 * m + 1).stack_rows(block.pad_right(F.dim()));
        weights =
            Tensor::concat({weights, full_block(like, 1, 1, 0.5), zeros_block(like, 1, 4 * m)}, -1);
        Gs.push_back(Tensor::concat({S.c().expand_dims(-1) * 0.5, S.generators().matrix_transpose(),
                                     zeros_block(like, n, 3 * m)},
                                    -1));
        c = c + S.c().expand_dims(-1) * 0.5;
    }
    // the weights of all sets add to one
    const Polytope sum =
        Polytope::equalities(weights, full_block(like, 1, 1, 1.0 - 0.5 * double(N)).squeeze(-1));
    return {c.squeeze(-1), Tensor::concat(Gs, -1).matrix_transpose(), F.stack_rows(sum)};
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
    CORA_CHECK(is_vector(p, dim()), Error::dimension_mismatch, [&] {
        // a point is a set without generators
        return "convex_hull: the point must be a vector (" + std::to_string(dim()) +
               "), got the shape " + shape_text(p.shape());
    });
    const detail::Refusal refusal = aux_refusal(*this, dim(), method);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_hull(*this, ConZonotope(p, zeros_block(p.expand_dims(-1), 0, dim())), method);
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
