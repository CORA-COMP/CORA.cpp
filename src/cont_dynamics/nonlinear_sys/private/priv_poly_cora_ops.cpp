// priv_poly_cora_ops - CORA's polynomial zonotope operations as CORALean's float layer has them
//
// The enclosure of two sets of the same factors (CORALean's encloseShared, then CORA's constructor:
// zero generators dropped, equal exponents added, in CORA's order), whose independent part is the
// linear combination of the two (PolyZonotope::enclose takes CORA's zonotope enclose), and
// reduce(pZ, 'girard', o) (reduceCora: the generators ranked by their length with the even ones
// halved, the shortest past (o - 1) n boxed into the independent part by the zonotope reduction to
// order 1; the zero independent generators stay, which the constructor of PolyZonotope drops).
//
// Syntax:   PolyZonotope E = priv_poly_enclose_shared(A, B);
//           PolyZonotope C = priv_poly_construct(Z);
//           PolyZonotope R = priv_poly_reduce_cora(Z, order);
// See also: priv_poly_ops.cpp, priv_lin_reach_poly.cpp
//
// CORA: contSet/@polyZonotope/enclose.m
// CORA: contSet/@polyZonotope/polyZonotope.m
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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The squared length of each of the h dependent (an even one halved) and q independent
/// generators, added over the coordinates from the first, and their order: the longest first.
std::vector<std::size_t> aux_ranked(const PolyZonotope &Z) {
    const int64_t n = Z.dim(), h = Z.num_dependent(), q = Z.independent_generators().shape()[1];
    const Vec g = Z.generators().data(), gi = Z.independent_generators().data();
    std::vector<std::pair<double, std::size_t>> rank(std::size_t(h + q));
    for (int64_t r = 0; r < h + q; ++r) {
        const bool even = r < h && priv_is_even(Z.exponents()[std::size_t(r)]);
        double len = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double x = r >= h ? gi[std::size_t(i * q + (r - h))]
                                    : g[std::size_t(i * h + r)] * (even ? 0.5 : 1.0);
            len = x * x + len;
        }
        rank[std::size_t(r)] = {len, std::size_t(r)};
    }
    // equal lengths stay in order
    std::sort(rank.begin(), rank.end(), [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::vector<std::size_t> order;
    order.reserve(rank.size());
    for (const auto &entry : rank) order.push_back(entry.second);
    return order;
}

/// The set of the dependent generators dep and the independent ones ind of Z, with its center or
/// the origin.
PolyZonotope aux_pick(const PolyZonotope &Z, const std::vector<int64_t> &dep,
                      const std::vector<int64_t> &ind, bool with_center) {
    const int64_t n = Z.dim();
    Exponents E;
    for (const int64_t l : dep) E.push_back(Z.exponents()[std::size_t(l)]);
    const auto columns = [&](const Tensor &M, const std::vector<int64_t> &idx) {
        return idx.empty() ? M.zeros_like({n, 0}) : priv_columns(M, idx);
    };
    return {with_center ? Z.center() : Z.center().zeros_like(),
            columns(Z.generators(), dep),
            columns(Z.independent_generators(), ind),
            std::move(E),
            Z.ids(),
            detail::Verbatim{}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constructor and the enclosure ---------------------------------------------------------------

PolyZonotope priv_poly_construct(const PolyZonotope &Z) {
    MergedColumns m = priv_merge_columns(Z.generators(), Z.exponents());
    return {Z.center(),     std::move(m.G), Z.independent_generators(),
            std::move(m.E), Z.ids(),        detail::Verbatim{}};
}

PolyZonotope priv_poly_enclose_shared(const PolyZonotope &A, const PolyZonotope &B) {
    // the midpoints and half-differences of the generators and of the centers
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
    // the independent parts as zonotopes of the center 0, combined
    const Tensor zero = A.center().zeros_like();
    const Zonotope Z = priv_lin_comb(Zonotope(zero, A.independent_generators()),
                                     Zonotope(zero, B.independent_generators()));
    Tensor GI = Z.generators();
    const int64_t q = A.independent_generators().shape()[1];
    // as many generators in both: [(G1 + G2)/2, (c1 - c2)/2, (G1 - G2)/2], the center in the middle
    if (q == B.independent_generators().shape()[1])
        GI = Tensor::concat(
            {GI.slice(-1, 0, q), GI.slice(-1, 2 * q, 2 * q + 1), GI.slice(-1, q, 2 * q)}, -1);
    return priv_poly_construct(
        {(A.center() + B.center()) * 0.5, G, GI, std::move(E), std::move(id), detail::Verbatim{}});
}

// The reduction ----------------------------------------------------------------------------------

PolyZonotope priv_poly_reduce_cora(const PolyZonotope &Z, int order) {
    const int64_t n = Z.dim(), h = Z.num_dependent(), q = Z.independent_generators().shape()[1];
    PolyZonotope out = Z;
    if (int64_t(order) * n < h + q) {
        // the removed generators past (order - 1) n of the ranking, the kept ones in their order
        const std::vector<std::size_t> ranked = aux_ranked(Z);
        std::vector<int64_t> removed_dep, removed_ind, kept_dep, kept_ind;
        std::vector<char> gone(std::size_t(h + q), 0);
        for (std::size_t t = std::size_t((order - 1) * n); t < ranked.size(); ++t) {
            const int64_t r = int64_t(ranked[t]);
            gone[std::size_t(r)] = 1;
            (r < h ? removed_dep : removed_ind).push_back(r < h ? r : r - h);
        }
        for (int64_t l = 0; l < h; ++l)
            if (!gone[std::size_t(l)]) kept_dep.push_back(l);
        for (int64_t j = 0; j < q; ++j)
            if (!gone[std::size_t(h + j)]) kept_ind.push_back(j);
        // the removed ones boxed by the zonotope reduction to order 1, joined to the kept ones
        const Zonotope boxed =
            priv_poly_construct(aux_pick(Z, removed_dep, removed_ind, false)).zonotope().reduce(1);
        out = priv_poly_plus(aux_pick(Z, kept_dep, kept_ind, true), boxed);
    }
    // the factors no generator reads go
    Exponents E = out.exponents();
    std::vector<int64_t> id = out.ids();
    priv_drop_unused_ids(E, id);
    return {out.center(), out.generators(), out.independent_generators(),
            std::move(E), std::move(id),    detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
