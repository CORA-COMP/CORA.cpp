// reduce - a polynomial zonotope with fewer generators that holds it, as CORA's reduce
//
// If the generators (dependent and independent) number more than order * n, the order * n - n
// longest stay (an even dependent generator counts half its length). The others are enclosed by
// the zonotope of pZ.zonotope() of them, reduced to order 1 (n box generators) and added to the
// center and the independent generators. The factors that no generator uses any more go.
//
// Syntax:   S = pZ.reduce(order);
// Inputs:   order - the zonotope order, at least 1
// Outputs:  S - a polynomial zonotope with at most order * n generators
// See also: zonotope
//
// CORA: contSet/@polyZonotope/reduce.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the order must be at least 1, got " + std::to_string(order);
    });
    const int64_t n = dim(), h = num_dependent(), q = GI_.shape()[1], m = h + q;
    PolyZonotope out = *this;
    // too many generators: the shortest ones are enclosed by a zonotope
    if (m > n * order) {
        // the squared lengths, an even dependent generator at half its length
        const std::vector<double> g = Tensor::concat({G_, GI_}, -1).data();
        std::vector<double> len(static_cast<std::size_t>(m), 0.0);
        for (int64_t j = 0; j < m; ++j) {
            const double w = (j < h && priv_is_even(E_[std::size_t(j)])) ? 0.25 : 1.0;
            for (int64_t i = 0; i < n; ++i)
                len[std::size_t(j)] += w * g[std::size_t(i * m + j)] * g[std::size_t(i * m + j)];
        }
        // the order * n - n longest stay (a stable sort, as MATLAB's), the others are boxed
        std::vector<int64_t> by_length(static_cast<std::size_t>(m));
        std::iota(by_length.begin(), by_length.end(), 0);
        std::stable_sort(by_length.begin(), by_length.end(), [&](int64_t a, int64_t b) {
            return len[std::size_t(a)] > len[std::size_t(b)];
        });
        std::vector<char> boxed(static_cast<std::size_t>(m), 0);
        std::vector<int64_t> boxed_dep, boxed_ind, kept_dep, kept_ind;
        // the generators after the first order * n - n by length are boxed
        for (std::size_t k = std::size_t(n * order - n); k < by_length.size(); ++k) {
            boxed[std::size_t(by_length[k])] = 1;
            if (by_length[k] < h)
                boxed_dep.push_back(by_length[k]);
            else
                boxed_ind.push_back(by_length[k] - h);
        }
        for (int64_t j = 0; j < h; ++j)
            if (!boxed[std::size_t(j)]) kept_dep.push_back(j);
        for (int64_t j = 0; j < q; ++j)
            if (!boxed[std::size_t(h + j)]) kept_ind.push_back(j);
        const auto exponents_of = [&](const std::vector<int64_t> &columns) {
            Exponents E;
            for (const int64_t j : columns) E.push_back(E_[std::size_t(j)]);
            return E;
        };
        // the boxed generators as a zonotope of order 1: its center and generators join
        const PolyZonotope rest(c_.zeros_like(), priv_columns(G_, boxed_dep),
                                priv_columns(GI_, boxed_ind), exponents_of(boxed_dep), id_);
        const Zonotope box = rest.zonotope().reduce(1);
        out = PolyZonotope(c_ + box.center(), priv_columns(G_, kept_dep),
                           Tensor::concat({priv_columns(GI_, kept_ind), box.generators()}, -1),
                           exponents_of(kept_dep), id_);
    }
    // the factors that no generator uses go
    Exponents E = out.exponents();
    std::vector<int64_t> id = out.ids();
    priv_drop_unused_ids(E, id);
    return {out.center(), out.generators(), out.independent_generators(), std::move(E),
            std::move(id)};
}

Set PolyZonotope::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
