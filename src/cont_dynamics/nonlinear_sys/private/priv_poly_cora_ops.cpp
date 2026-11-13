// priv_poly_cora_ops - CORA's operations on polynomial zonotopes, as CORALean's float layer has
// them
//
// The enclosure of two sets of the same factors (encloseShared), CORA's constructor (zero
// generators dropped, equal exponents added, in the order of the first occurrences), zonotope(pZ)
// (zonotopeFold: the even generators halved, the constant ones and the other halves added into the
// center from the last generator to the first) and reduce(pZ, 'girard', o) (reduceCora: the
// generators ranked by their length with the even ones halved, the shortest past (o - 1) n boxed
// into the independent part by the zonotope reduction to order 1). The sums add from the last term
// to the first as CORALean's sumDown, the arithmetic is nearest.
//
// Syntax:   PolyZonotope E = priv_poly_enclose_shared(A, B);
//           PolyZonotope C = priv_poly_construct(Z);
//           Zonotope F = priv_poly_zonotope_fold(Z);
//           PolyZonotope R = priv_poly_reduce_cora(Z, order);
// See also: priv_poly_ops.cpp, priv_lin_reach_poly.cpp
//
// CORA: contSet/@polyZonotope/enclose.m
// CORA: contSet/@polyZonotope/polyZonotope.m
// CORA: contSet/@polyZonotope/zonotope.m
// CORA: contSet/@polyZonotope/reduce.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"
#include "cont_set/poly_zonotope/private/priv.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ===========================================  MAIN  =========================================== //

/// The enclosure of the segments between the points of A and B, which have the factors and the
/// exponents of A (CORALean's encloseShared): the midpoints and half-differences of the
/// generators and of the centers, on the factors of A and one new one; the independent parts are
/// those of the linear combination, with the column of the center in the middle when both have as
/// many generators.
PolyZonotope priv_poly_enclose_shared(const PolyZonotope &A, const PolyZonotope &B) {
    const Tensor &like = A.center();
    const Tensor G =
        Tensor::concat({(A.generators() + B.generators()) * 0.5,
                        (A.generators() - B.generators()) * 0.5, (A.center() - B.center()) * 0.5},
                       -1);
    // the exponents: the new factor 0 on the first block, 1 on the second and on the center column
    Exponents E;
    for (int block = 0; block < 2; ++block)
        for (const std::vector<int> &col : A.exponents()) {
            std::vector<int> c = col;
            c.push_back(block);
            E.push_back(std::move(c));
        }
    std::vector<int> last(A.ids().size() + 1, 0);
    last.back() = 1;
    E.push_back(std::move(last));
    std::vector<int64_t> id = A.ids();
    id.push_back((id.empty() ? 0 : *std::max_element(id.begin(), id.end())) + 1);
    // the independent parts as zonotopes of the center 0
    const Tensor zero = like.zeros_like();
    const Zonotope Z = priv_lin_comb(Zonotope(zero, A.independent_generators()),
                                     Zonotope(zero, B.independent_generators()));
    Tensor GI = Z.generators();
    const int64_t q1 = A.independent_generators().shape()[1];
    if (q1 == B.independent_generators().shape()[1]) {
        // [(G1 + G2)/2, (c1 - c2)/2, (G1 - G2)/2]: the column of the center after the first q1
        GI = Tensor::concat(
            {GI.slice(-1, 0, q1), GI.slice(-1, 2 * q1, 2 * q1 + 1), GI.slice(-1, q1, 2 * q1)}, -1);
    }
    return {
        (A.center() + B.center()) * 0.5, G, GI, std::move(E), std::move(id), detail::Verbatim{}};
}

/// CORA's constructor, as CORALean's construct in the order of the first occurrences: the zero
/// generators dropped, then the generators of equal exponents added.
PolyZonotope priv_poly_construct(const PolyZonotope &Z) {
    const int64_t n = Z.dim(), h = Z.num_dependent();
    const Vec g = Z.generators().data();
    std::vector<int64_t> live;
    for (int64_t l = 0; l < h; ++l) {
        bool zero = true;
        for (int64_t i = 0; i < n; ++i) zero = zero && g[std::size_t(i * h + l)] == 0.0;
        if (!zero) live.push_back(l);
    }
    Exponents E;
    for (const int64_t l : live) E.push_back(Z.exponents()[std::size_t(l)]);
    const Tensor G = Tensor::like(Z.generators(),
                                  [&] {
                                      Vec out(std::size_t(n) * live.size());
                                      for (int64_t i = 0; i < n; ++i)
                                          for (std::size_t t = 0; t < live.size(); ++t)
                                              out[std::size_t(i) * live.size() + t] =
                                                  g[std::size_t(i * h + live[t])];
                                      return out;
                                  }(),
                                  {n, int64_t(live.size())});
    return PolyZonotope(Z.center(), G, Z.independent_generators(), std::move(E), Z.ids(),
                        detail::Verbatim{})
        .compact_gens();
}

/// CORA's zonotope(pZ), as CORALean's zonotopeFold: the generators that are not all even, the
/// even ones halved, the independent ones; the constant ones and the other half of the even ones
/// go into the center, each sum added from the last generator to the first.
Zonotope priv_poly_zonotope_fold(const PolyZonotope &Z) {
    const int64_t n = Z.dim(), h = Z.num_dependent();
    const Vec g = Z.generators().data();
    std::vector<int64_t> odd, even;
    Vec shift(std::size_t(n), 0.0);
    std::vector<char> is_constant(std::size_t(h), 0), is_even(std::size_t(h), 0);
    for (int64_t l = h; l-- > 0;) {
        const std::vector<int> &col = Z.exponents()[std::size_t(l)];
        const bool constant = std::all_of(col.begin(), col.end(), [](int e) { return e == 0; });
        const bool all_even = std::all_of(col.begin(), col.end(), [](int e) { return e % 2 == 0; });
        is_constant[std::size_t(l)] = constant;
        is_even[std::size_t(l)] = all_even && !constant;
        for (int64_t i = 0; i < n; ++i) {
            const double x = g[std::size_t(i * h + l)];
            const double t = constant ? x : (all_even ? x * 0.5 : 0.0);
            shift[std::size_t(i)] = t + shift[std::size_t(i)];
        }
    }
    for (int64_t l = 0; l < h; ++l) {
        if (is_constant[std::size_t(l)]) continue;
        (is_even[std::size_t(l)] ? even : odd).push_back(l);
    }
    const int64_t q = Z.independent_generators().shape()[1];
    const std::size_t m = odd.size() + even.size() + std::size_t(q);
    const Vec gi = Z.independent_generators().data();
    Vec out(std::size_t(n) * m);
    for (int64_t i = 0; i < n; ++i) {
        std::size_t col = 0;
        for (const int64_t l : odd) out[std::size_t(i) * m + col++] = g[std::size_t(i * h + l)];
        for (const int64_t l : even)
            out[std::size_t(i) * m + col++] = g[std::size_t(i * h + l)] * 0.5;
        for (int64_t j = 0; j < q; ++j)
            out[std::size_t(i) * m + col++] = gi[std::size_t(i * q + j)];
    }
    return {Z.center() + Tensor::like(Z.center(), shift, {n, 1}),
            Tensor::like(Z.center(), out, {n, int64_t(m)})};
}


PolyZonotope priv_poly_reduce_cora(const PolyZonotope &Z, int order) {
    const int64_t n = Z.dim(), h = Z.num_dependent(), q = Z.independent_generators().shape()[1];
    PolyZonotope out = Z;
    if (int64_t(order) * n < h + q) {
        const Vec g = Z.generators().data(), gi = Z.independent_generators().data();
        // the squared length of each generator, an even dependent one halved, added over the
        // coordinates from the first
        std::vector<std::pair<double, std::size_t>> rank(std::size_t(h + q));
        for (int64_t r = 0; r < h + q; ++r) {
            const std::vector<int> *col = r < h ? &Z.exponents()[std::size_t(r)] : nullptr;
            const bool even =
                col && std::all_of(col->begin(), col->end(), [](int e) { return e % 2 == 0; });
            double len = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double x =
                    r < h ? (even ? g[std::size_t(i * h + r)] * 0.5 : g[std::size_t(i * h + r)])
                          : gi[std::size_t(i * q + (r - h))];
                len = x * x + len;
            }
            rank[std::size_t(r)] = {len, std::size_t(r)};
        }
        // the longest first, equal lengths in order
        std::sort(rank.begin(), rank.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        // the removed ones past (order - 1) n, in rank; the kept ones in their order
        std::vector<int64_t> rD, rI, kD, kI;
        std::vector<char> gone(std::size_t(h + q), 0);
        for (std::size_t t = std::size_t((order - 1) * n); t < rank.size(); ++t) {
            const int64_t r = int64_t(rank[t].second);
            gone[std::size_t(r)] = 1;
            (r < h ? rD : rI).push_back(r < h ? r : r - h);
        }
        for (int64_t l = 0; l < h; ++l)
            if (!gone[std::size_t(l)]) kD.push_back(l);
        for (int64_t j = 0; j < q; ++j)
            if (!gone[std::size_t(h + j)]) kI.push_back(j);
        const auto pick = [&](const std::vector<int64_t> &dep, const std::vector<int64_t> &ind,
                              bool with_center) {
            Exponents E;
            for (const int64_t l : dep) E.push_back(Z.exponents()[std::size_t(l)]);
            const auto columns = [&](const Tensor &M, const std::vector<int64_t> &idx) {
                return idx.empty() ? M.zeros_like({n, 0}) : priv_columns(M, idx);
            };
            return PolyZonotope(with_center ? Z.center() : Z.center().zeros_like(),
                                columns(Z.generators(), dep),
                                columns(Z.independent_generators(), ind), std::move(E), Z.ids(),
                                detail::Verbatim{});
        };
        const Zonotope Zr =
            priv_poly_zonotope_fold(priv_poly_construct(pick(rD, rI, false))).reduce(1);
        out = priv_poly_plus(pick(kD, kI, true), Zr);
    }
    // the factors no generator reads go
    Exponents E = out.exponents();
    std::vector<int64_t> id = out.ids();
    priv_drop_unused_ids(E, id);
    return PolyZonotope(out.center(), out.generators(), out.independent_generators(), std::move(E),
                        std::move(id), detail::Verbatim{});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
