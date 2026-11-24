// priv_exponents - the exponent matrices and identifiers of polynomial zonotopes
//
// See also: priv.h
//
// CORA: global/functions/helper/sets/contSet/polyZonotope/removeRedundantExponents.m
// CORA: global/functions/helper/sets/contSet/polyZonotope/mergeExpMatrix.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The hash vector ---------------------------------------------------------------------------

std::vector<double> priv_hash_vector(std::size_t p) {
    // rand(1, p) of RandStream('mt19937ar', 'Seed', 0): std::mt19937 seeded 5489, two words each
    static const std::vector<double> table = [] {
        std::mt19937 gen(5489);
        std::vector<double> v(256);
        for (double &x : v) {
            const uint32_t a = gen() >> 5, b = gen() >> 6;
            x = (double(a) * 67108864.0 + double(b)) / 9007199254740992.0;
        }
        return v;
    }();
    if (p <= table.size()) return {table.begin(), table.begin() + std::ptrdiff_t(p)};
    std::mt19937 gen(5489);
    std::vector<double> v(p);
    for (double &x : v) {
        const uint32_t a = gen() >> 5, b = gen() >> 6;
        x = (double(a) * 67108864.0 + double(b)) / 9007199254740992.0;
    }
    return v;
}

// The columns -----------------------------------------------------------------------------------

std::vector<int64_t> priv_least_equal_column(const Exponents &E) {
    // an open-addressing table of column indices, compared by their exponents
    const std::size_t h = E.size();
    std::size_t capacity = 16;
    while (capacity < 2 * h) capacity *= 2;
    std::vector<int64_t> table(capacity, -1), rep(h);
    for (std::size_t j = 0; j < h; ++j) {
        std::size_t hash = 1469598103934665603ULL; // FNV-1a over the exponents
        for (const int e : E[j]) hash = (hash ^ static_cast<std::size_t>(e)) * 1099511628211ULL;
        // probe from the hash to the first column equal to this one, or to a free slot
        std::size_t slot = (hash ^ (hash >> 29)) & (capacity - 1);
        while (table[slot] >= 0 && E[std::size_t(table[slot])] != E[j])
            slot = (slot + 1) & (capacity - 1);
        if (table[slot] < 0) table[slot] = int64_t(j);
        rep[j] = table[slot];
    }
    return rep;
}

Tensor priv_rows(const Tensor &G, const std::vector<int64_t> &index) {
    const int64_t n = G.shape()[1];
    if (index.empty()) return G.zeros_like({0, n});
    // all rows in their order: the tensor itself
    if (int64_t(index.size()) == G.shape()[0]) {
        int64_t next = 0;
        if (std::all_of(index.begin(), index.end(), [&](int64_t j) { return j == next++; }))
            return G;
    }
    return G.take(index, -2);
}

bool priv_is_even(const std::vector<int> &exponents) {
    return std::all_of(exponents.begin(), exponents.end(), [](int e) { return e % 2 == 0; });
}

MergedColumns priv_merge_columns(const Tensor &G, const Exponents &E) {
    const int64_t h = G.shape()[0], n = G.shape()[1];
    if (h == 0) return {G, E};
    const std::vector<double> g = G.data();
    // the columns that are not zero, with a hash of their exponents
    std::vector<int64_t> live;
    std::vector<std::size_t> hash(std::size_t(h), 0);
    bool changed = false;
    for (int64_t j = 0; j < h; ++j) {
        bool nonzero = false;
        for (int64_t i = 0; i < n && !nonzero; ++i) nonzero = g[std::size_t(j * n + i)] != 0.0;
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
    bool repeated = false;
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
                repeated = true;
            first[std::size_t(j)] = lead;
        }
        r = end;
    }
    if (!repeated && !changed) return {G, E};
    // no two columns equal: CORA returns the columns as they are, the zero ones removed
    if (!repeated) {
        Exponents kept;
        for (const int64_t j : live) kept.push_back(E[std::size_t(j)]);
        return {priv_rows(G, live), std::move(kept)};
    }
    // the groups, the columns of each in order: the leaders by their first column
    std::vector<int64_t> leaders;
    std::vector<std::vector<int64_t>> members;
    std::vector<std::size_t> slot(std::size_t(h), 0);
    for (const int64_t j : live) {
        if (first[std::size_t(j)] == j) {
            slot[std::size_t(j)] = leaders.size();
            leaders.push_back(j);
            members.emplace_back();
        }
        members[slot[std::size_t(first[std::size_t(j)])]].push_back(j);
    }
    // CORA: columns of a single exponent vector first, then the sums, each by hash (stable)
    const std::vector<double> hash_vec = priv_hash_vector(E[std::size_t(live[0])].size());
    std::vector<double> key(leaders.size());
    for (std::size_t t = 0; t < leaders.size(); ++t) {
        double sum = 0.0;
        const std::vector<int> &col = E[std::size_t(leaders[t])];
        for (std::size_t i = 0; i < col.size(); ++i) {
            const double term = hash_vec[i] * double(col[i]);
            sum = i == 0 ? term : sum + term;
        }
        key[t] = sum;
    }
    std::vector<std::size_t> rank(leaders.size());
    std::iota(rank.begin(), rank.end(), std::size_t(0));
    // ties by the place of the first column, which makes the sort stable
    std::sort(rank.begin(), rank.end(), [&](std::size_t a, std::size_t b) {
        const bool multiple_a = members[a].size() > 1, multiple_b = members[b].size() > 1;
        if (multiple_a != multiple_b) return !multiple_a;
        return key[a] != key[b] ? key[a] < key[b] : a < b;
    });
    // a group adds up from its last column to the first (CORALean's compactGens, not MATLAB's)
    const auto k = int64_t(leaders.size());
    std::vector<double> merged(std::size_t(n * k), 0.0);
    Exponents E_out;
    for (std::size_t t = 0; t < rank.size(); ++t) {
        const std::vector<int64_t> &group = members[rank[t]];
        E_out.push_back(E[std::size_t(leaders[rank[t]])]);
        for (int64_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t m = group.size(); m-- > 0;)
                sum = g[std::size_t(group[m] * n + i)] + sum;
            merged[t * std::size_t(n) + std::size_t(i)] = sum;
        }
    }
    // sums that cancelled (x and -x) go too, as the second pass of CORA's constructor removes them
    std::vector<int64_t> nonzero;
    for (int64_t t = 0; t < k; ++t)
        for (int64_t i = 0; i < n; ++i)
            if (merged[std::size_t(t * n + i)] != 0.0) {
                nonzero.push_back(t);
                break;
            }
    const Tensor all = Tensor::like(G, merged, {k, n});
    if (int64_t(nonzero.size()) == k) return {all, std::move(E_out)};
    Exponents kept;
    for (const int64_t t : nonzero) kept.push_back(E_out[std::size_t(t)]);
    return {priv_rows(all, nonzero), std::move(kept)};
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
    const Tensor c = set.center() + priv_rows(set.generators(), constant).sum(-2);
    return {c,
            priv_rows(set.generators(), rest),
            set.independent_generators(),
            std::move(E),
            set.ids(),
            detail::Normalized{}};
}

PolyZonotope priv_merged_set(const Tensor &c, const Tensor &G, const Tensor &GI, const Exponents &E,
                             const std::vector<int64_t> &id) {
    // equal columns merge first, as CORA does, then the constants and the unused identifiers go
    MergedColumns merged = priv_merge_columns(G, E);
    const PolyZonotope set(c, std::move(merged.G), GI, std::move(merged.E), id,
                           detail::Normalized{});
    // the center takes the constants, the unused identifiers leave the exponents
    const PolyZonotope folded = priv_fold_constants(set);
    Exponents E_out = folded.exponents();
    std::vector<int64_t> id_out = folded.ids();
    // the unused identifiers leave
    priv_drop_unused_ids(E_out, id_out);
    return {folded.center(),  folded.generators(), folded.independent_generators(),
            std::move(E_out), std::move(id_out),   detail::Normalized{}};
}

std::vector<int64_t> priv_append_fresh_ids(const std::vector<int64_t> &ids, int64_t count) {
    // the new identifiers count on from the largest one
    std::vector<int64_t> out = ids;
    const int64_t top = ids.empty() ? int64_t(0) : *std::max_element(ids.begin(), ids.end());
    for (int64_t j = 1; j <= count; ++j) out.push_back(top + j);
    return out;
}

Exponents priv_extended_exponents(const Exponents &E, std::size_t rows) {
    // the zero column of the center first
    Exponents out;
    out.reserve(E.size() + 1);
    out.emplace_back(rows, 0);
    out.insert(out.end(), E.begin(), E.end());
    return out;
}

PolyZonotope priv_absorb_rounding(const PolyZonotope &set) {
    const int64_t n = set.dim(), q = set.independent_generators().shape()[0], k = q + n;
    const std::size_t p = set.ids().size();
    // the old columns padded with the new factors, then one unit column per new factor
    Exponents E;
    for (const std::vector<int> &col : set.exponents()) {
        std::vector<int> e = col;
        e.resize(p + std::size_t(k), 0);
        E.push_back(std::move(e));
    }
    // the unit columns of the new factors
    for (int64_t j = 0; j < k; ++j) {
        std::vector<int> e(p + std::size_t(k), 0);
        e[p + std::size_t(j)] = 1;
        E.push_back(std::move(e));
    }
    // the generators, the independent ones, then the zero columns of the error box
    const Tensor G = Tensor::concat(
        {set.generators(), set.independent_generators(), set.center().zeros_like({n, n})}, -2);
    return {set.center(),
            G,
            set.center().zeros_like({0, n}),
            std::move(E),
            priv_append_fresh_ids(set.ids(), k),
            detail::Verbatim{}};
}


Tensor priv_pair_independent(const Tensor &A, const Tensor &B) {
    // the paired rows, then the surplus of the longer block
    const int64_t qa = A.shape()[0], qb = B.shape()[0], common = std::min(qa, qb);
    std::vector<Tensor> sums{(A.slice(-2, 0, common) + B.slice(-2, 0, common)) * 0.5};
    std::vector<Tensor> diffs{(A.slice(-2, 0, common) - B.slice(-2, 0, common)) * 0.5};
    if (qa > common) sums.push_back(A.slice(-2, common, qa));
    if (qb > common) diffs.push_back(B.slice(-2, common, qb));
    sums.insert(sums.end(), diffs.begin(), diffs.end());
    return Tensor::concat(sums, -2);
}

PolyZonotope priv_multiply(const PolyZonotope &a, const PolyZonotope &b) {
    const Tensor ga = Tensor::concat({a.center().expand_dims(0), a.generators()}, -2);
    const Tensor gb = Tensor::concat({b.center().expand_dims(0), b.generators()}, -2);
    const int64_t na = ga.shape()[0], nb = gb.shape()[0], n = a.dim();
    // every pair, the first column the slower index
    const Tensor G = ga.reshape({na, 1, n}).multiply(gb.reshape({1, nb, n})).reshape({na * nb, n});
    // the exponents of the pairs, the first column the slower index
    const Exponents Ea = priv_extended_exponents(a.exponents(), a.ids().size());
    const Exponents Eb = priv_extended_exponents(b.exponents(), b.ids().size());
    Exponents E;
    E.reserve(Ea.size() * Eb.size());
    // the sums of the exponent columns
    for (const std::vector<int> &x : Ea)
        for (const std::vector<int> &y : Eb) {
            std::vector<int> e = x;
            for (std::size_t i = 0; i < e.size(); ++i) e[i] += y[i];
            E.push_back(std::move(e));
        }
    return {a.center().zeros_like(), G, a.center().zeros_like({0, n}), std::move(E), a.ids(),
            detail::Verbatim{}};
}

int64_t priv_find_id(const std::vector<int64_t> &ids, int64_t id) {
    const auto it = std::find(ids.begin(), ids.end(), id);
    return it == ids.end() ? -1 : int64_t(it - ids.begin());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
