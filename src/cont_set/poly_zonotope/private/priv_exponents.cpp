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
#include <unordered_map>

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
    const std::vector<double> g = G.data();
    // the columns that are not zero, with a hash of their exponents
    std::vector<int64_t> live;
    std::vector<std::size_t> hash(std::size_t(h), 0);
    bool changed = false;
    for (int64_t j = 0; j < h; ++j) {
        bool nonzero = false;
        for (int64_t i = 0; i < n && !nonzero; ++i) nonzero = g[std::size_t(i * h + j)] != 0.0;
        if (!nonzero) {
            changed = true;
            continue;
        }
        std::size_t x = 1469598103934665603ULL;
        for (const int e : E[std::size_t(j)]) x = (x ^ std::size_t(e)) * 1099511628211ULL;
        hash[std::size_t(j)] = x;
        live.push_back(j);
    }
    // group the live columns of equal exponents: sorted by (hash, index), runs verified
    std::vector<int64_t> order = live;
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return hash[std::size_t(a)] != hash[std::size_t(b)]
                   ? hash[std::size_t(a)] < hash[std::size_t(b)]
                   : a < b;
    });
    std::vector<int64_t> first(std::size_t(h), -1); // the first column of the group of j
    for (std::size_t r = 0; r < order.size();) {
        std::size_t end = r;
        while (end < order.size() && hash[std::size_t(order[end])] == hash[std::size_t(order[r])])
            ++end;
        std::vector<int64_t> leaders; // the distinct exponents of this run
        for (std::size_t t = r; t < end; ++t) {
            const int64_t j = order[t];
            int64_t lead = j;
            for (const int64_t l : leaders)
                if (E[std::size_t(l)] == E[std::size_t(j)]) {
                    lead = l;
                    break;
                }
            if (lead == j)
                leaders.push_back(j);
            else
                changed = true;
            first[std::size_t(j)] = lead;
        }
        r = end;
    }
    if (!changed) return {G, E};
    // the groups in the order of their first column, the columns of a group added in order
    std::vector<int64_t> slot(std::size_t(h), -1);
    Exponents kept;
    for (const int64_t j : live)
        if (first[std::size_t(j)] == j) {
            slot[std::size_t(j)] = int64_t(kept.size());
            kept.push_back(E[std::size_t(j)]);
        }
    const auto k = int64_t(kept.size());
    if (k == 0) return {G.zeros_like({n, 0}), {}};
    std::vector<double> merged(std::size_t(n * k), 0.0);
    for (const int64_t j : live) {
        const auto t = std::size_t(slot[std::size_t(first[std::size_t(j)])]);
        for (int64_t i = 0; i < n; ++i)
            merged[std::size_t(i) * std::size_t(k) + t] += g[std::size_t(i * h + j)];
    }
    // generators that cancel in the sum (x and -x) go too
    std::vector<int64_t> nonzero;
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i)
            if (merged[std::size_t(i * k + j)] != 0.0) {
                nonzero.push_back(j);
                break;
            }
    const Tensor all = Tensor::like(G, merged, {n, k});
    if (int64_t(nonzero.size()) == k) return {all, std::move(kept)};
    Exponents E_out;
    for (const int64_t j : nonzero) E_out.push_back(kept[std::size_t(j)]);
    return {priv_columns(all, nonzero), std::move(E_out)};
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
    return {c,
            priv_columns(set.generators(), rest),
            set.independent_generators(),
            std::move(E),
            set.ids(),
            detail::Normalized{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
