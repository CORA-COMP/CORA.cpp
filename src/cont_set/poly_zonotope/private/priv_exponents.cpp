// priv_exponents - the exponent matrices and identifiers of polynomial zonotopes
//
// See also: priv.h
//
// CORA: global/functions/helper/sets/contSet/polyZonotope/removeRedundantExponents.m
// CORA: global/functions/helper/sets/contSet/polyZonotope/mergeExpMatrix.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

#include <algorithm>
#include <map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The columns -----------------------------------------------------------------------------------

Tensor priv_columns(const Tensor &G, const std::vector<int64_t> &index) {
    const int64_t n = G.shape()[0];
    if (index.empty()) return G.zeros_like({n, 0});
    return G.take(index, -1);
}

bool priv_is_even(const std::vector<int> &exponents) {
    return std::all_of(exponents.begin(), exponents.end(), [](int e) { return e % 2 == 0; });
}

MergedColumns priv_merge_columns(const Tensor &G, const Exponents &E) {
    const int64_t n = G.shape()[0], h = G.shape()[1];
    if (h == 0) return {G, E};
    // the columns of G that are not zero, grouped by their exponents (first occurrence first)
    const std::vector<double> g = G.data();
    std::map<std::vector<int>, int64_t> slot;
    std::vector<int64_t> target(std::size_t(h), -1);
    Exponents kept;
    bool changed = false;
    for (int64_t j = 0; j < h; ++j) {
        bool nonzero = false;
        for (int64_t i = 0; i < n && !nonzero; ++i) nonzero = g[std::size_t(i * h + j)] != 0.0;
        if (!nonzero) {
            changed = true;
            continue;
        }
        const auto [it, fresh] = slot.emplace(E[std::size_t(j)], int64_t(kept.size()));
        if (fresh)
            kept.push_back(E[std::size_t(j)]);
        else
            changed = true;
        target[std::size_t(j)] = it->second;
    }
    if (!changed) return {G, E};
    // G times the 0/1 matrix that adds the columns of one group
    const auto k = int64_t(kept.size());
    if (k == 0) return {G.zeros_like({n, 0}), {}};
    std::vector<double> S(std::size_t(h * k), 0.0);
    for (int64_t j = 0; j < h; ++j)
        if (target[std::size_t(j)] >= 0) S[std::size_t(j * k + target[std::size_t(j)])] = 1.0;
    const Tensor merged = G.matmul(Tensor::like(G, S, {h, k}));
    // generators that cancel in the sum (x and -x) go too
    const std::vector<double> m = merged.data();
    std::vector<int64_t> nonzero;
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i)
            if (m[std::size_t(i * k + j)] != 0.0) {
                nonzero.push_back(j);
                break;
            }
    if (int64_t(nonzero.size()) == k) return {merged, std::move(kept)};
    Exponents E_out;
    for (const int64_t j : nonzero) E_out.push_back(kept[std::size_t(j)]);
    return {priv_columns(merged, nonzero), std::move(E_out)};
}

// The identifiers -------------------------------------------------------------------------------

MergedIds priv_merge_ids(const std::vector<int64_t> &id1, const std::vector<int64_t> &id2) {
    MergedIds out;
    out.id = id1;
    out.id.insert(out.id.end(), id2.begin(), id2.end());
    std::sort(out.id.begin(), out.id.end());
    out.id.erase(std::unique(out.id.begin(), out.id.end()), out.id.end());
    // where each identifier of a set sits in the union
    const auto place = [&](const std::vector<int64_t> &id) {
        std::vector<std::size_t> position;
        position.reserve(id.size());
        for (const int64_t v : id)
            position.push_back(
                std::size_t(std::lower_bound(out.id.begin(), out.id.end(), v) - out.id.begin()));
        return position;
    };
    out.position1 = place(id1);
    out.position2 = place(id2);
    return out;
}

Exponents priv_extend_exponents(const Exponents &E, const std::vector<std::size_t> &position,
                                std::size_t rows) {
    Exponents out(E.size(), std::vector<int>(rows, 0));
    for (std::size_t j = 0; j < E.size(); ++j)
        for (std::size_t i = 0; i < position.size(); ++i) out[j][position[i]] = E[j][i];
    return out;
}

// The constants ---------------------------------------------------------------------------------

void priv_drop_unused_ids(Exponents &E, std::vector<int64_t> &id) {
    // the identifiers with a nonzero exponent in some generator
    std::vector<std::size_t> used;
    for (std::size_t i = 0; i < id.size(); ++i)
        if (std::any_of(E.begin(), E.end(),
                        [&](const std::vector<int> &col) { return col[i] != 0; }))
            used.push_back(i);
    if (used.size() == id.size()) return;
    // the exponents of the others are removed from every column
    std::vector<int64_t> id_new;
    id_new.reserve(used.size());
    // the rows of the unused identifiers leave every column
    for (const std::size_t i : used) id_new.push_back(id[i]);
    for (std::vector<int> &col : E) {
        std::vector<int> col_new;
        col_new.reserve(used.size());
        for (const std::size_t i : used) col_new.push_back(col[i]);
        col = std::move(col_new);
    }
    id = std::move(id_new);
}

// A generator of exponent 0 does not depend on a factor: it belongs to the center.
PolyZonotope priv_fold_constants(const PolyZonotope &set) {
    std::vector<int64_t> constant, rest;
    for (int64_t j = 0; j < set.num_dependent(); ++j) {
        const std::vector<int> &col = set.exponents()[std::size_t(j)];
        (std::all_of(col.begin(), col.end(), [](int e) { return e == 0; }) ? constant : rest)
            .push_back(j);
    }
    if (constant.empty()) return set;
    // the others keep their exponents, the constants add up in the center
    Exponents E;
    for (const int64_t j : rest) E.push_back(set.exponents()[std::size_t(j)]);
    const Tensor c = set.center() + priv_columns(set.generators(), constant).sum(-1, true);
    return {c, priv_columns(set.generators(), rest), set.independent_generators(), std::move(E),
            set.ids()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
