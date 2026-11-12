// priv_poly_ops - reductions of polynomial zonotopes for poly-adaptive
//
// The functions are CORALean's float definitions of CORA's reduce(pZ, 'adaptive', diagpercent)
// (the generators ranked by their length, the longest prefix whose box distance stays within
// diagpercent boxed, an even-exponent generator at its full radius), aux_reduceOnlyDep (past
// (order + 1) n dependent generators the shortest are boxed), restructure (the independent part
// boxed and made dependent) and approxVolumeRatio. The boxed prefix of the ranking ends where the
// cumulative box distance first exceeds dHmax (the increments are not negative), so the rankings of
// the blocks are produced only that far (lazy_rank.h) and merged. Sums add from the last term to
// the first and a norm is the root of the sum of squares, as CORALean's sumUp and normUp at
// nearest.
//
// Syntax:   PolyZonotope R = priv_reduce_adaptive_poly(Z, diagpercent);
// Inputs:   Z - polynomial zonotope
// Outputs:  the reduced set
// See also: priv_poly_adaptive.h, priv_reach_poly_adaptive.cpp
//
// CORA: contSet/@polyZonotope/reduce.m
// CORA: contSet/@polyZonotope/private/priv_reduceAdaptive.m
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"
#include "cont_set/poly_zonotope/private/priv.h"
#include "global/lazy_rank.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The data of a polynomial zonotope on the host: G is (n, h) and GI (n, q), row-major.
struct Host {
    Host(const PolyZonotope &Z)
        : n(Z.dim()), h(Z.num_dependent()), q(Z.independent_generators().shape()[1]),
          c(Z.center().data()), G(Z.generators().data()), GI(Z.independent_generators().data()),
          E(Z.exponents()) {}

    int64_t n, h, q;
    Vec c, G, GI;
    const Exponents &E; // the exponents of the set, which outlives the view

    double g(int64_t i, int64_t l) const { return G[std::size_t(i * h + l)]; }
    double gi(int64_t i, int64_t l) const { return GI[std::size_t(i * q + l)]; }
};

/// The tensor (rows, cols) of the values, like `like`; a tensor without columns is empty.
Tensor aux_tensor(const Tensor &like, const Vec &data, int64_t rows, int64_t cols) {
    return cols == 0 ? like.zeros_like({rows, 0}) : Tensor::like(like, data, {rows, cols});
}

/// Whether all exponents of the dependent generator are even (a constant one included).
bool aux_all_even(const std::vector<int> &col) {
    return std::all_of(col.begin(), col.end(), [](int e) { return e % 2 == 0; });
}

/// The indices 0 .. m-1 sorted by the key, the smaller first, ties in index order. The pairs
/// (key, index) are sorted, which is the same total order as sorting the indices by the key with
/// the index as the tie-break (a stable order), without the indirection.
std::vector<std::size_t> aux_rank(const Vec &key, bool descending = false) {
    std::vector<std::pair<double, std::size_t>> pairs(key.size());
    for (std::size_t i = 0; i < key.size(); ++i) pairs[i] = {key[i], i};
    std::sort(pairs.begin(), pairs.end(), [descending](const auto &a, const auto &b) {
        if (a.first != b.first) return descending ? b.first < a.first : a.first < b.first;
        return a.second < b.second;
    });
    std::vector<std::size_t> idx(key.size());
    for (std::size_t i = 0; i < key.size(); ++i) idx[i] = pairs[i].second;
    return idx;
}

/// The set with equal exponent columns added (compact_gens), without a copy when there are none.
PolyZonotope aux_compact(PolyZonotope &&Z) {
    const std::vector<int64_t> rep = priv_least_equal_column(Z.exponents());
    for (std::size_t j = 0; j < rep.size(); ++j)
        if (rep[j] != int64_t(j)) return Z.compact_gens();
    return std::move(Z);
}

/// The set with the dependent generators `keep` (in that order) and the independent generators
/// `keep_i`, and one more independent generator per coordinate of the radius `drop`, which boxes
/// the others (CORALean's reduceBy).
PolyZonotope aux_reduce_by(const PolyZonotope &Z, const Host &a,
                           const std::vector<std::size_t> &keep,
                           const std::vector<std::size_t> &keep_i, const Vec &drop) {
    const Tensor &like = Z.center();
    const int64_t n = a.n, hk = int64_t(keep.size()), qk = int64_t(keep_i.size());
    Vec G(std::size_t(n * hk)), GI(std::size_t(n * (qk + n)), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t l = 0; l < hk; ++l)
            G[std::size_t(i * hk + l)] = a.g(i, int64_t(keep[std::size_t(l)]));
        for (int64_t l = 0; l < qk; ++l)
            GI[std::size_t(i * (qk + n) + l)] = a.gi(i, int64_t(keep_i[std::size_t(l)]));
        GI[std::size_t(i * (qk + n) + qk + i)] = drop[std::size_t(i)];
    }
    // factors no kept generator carries go with their identifiers (no value changes)
    std::vector<int64_t> id;
    std::vector<std::size_t> used;
    for (std::size_t k = 0; k < Z.ids().size(); ++k)
        if (std::any_of(keep.begin(), keep.end(), [&](std::size_t l) { return a.E[l][k] != 0; })) {
            used.push_back(k);
            id.push_back(Z.ids()[k]);
        }
    Exponents E;
    E.reserve(keep.size());
    for (const std::size_t l : keep) {
        std::vector<int> c;
        c.reserve(used.size());
        for (const std::size_t k : used) c.push_back(a.E[l][k]);
        E.push_back(std::move(c));
    }
    return {
        Z.center(),    aux_tensor(like, G, n, hk), aux_tensor(like, GI, n, qk + n), std::move(E),
        std::move(id), detail::Verbatim{}};
}

/// The radius per coordinate of the dependent generators `dep` and the independent ones `ind`: the
/// row sums of the magnitudes, added upward (CORALean's dropRadius).
Vec aux_drop_radius(const Host &a, const std::vector<std::size_t> &dep,
                    const std::vector<std::size_t> &ind) {
    Vec out(std::size_t(a.n));
    for (int64_t i = 0; i < a.n; ++i) {
        double d = 0, e = 0;
        for (std::size_t l = dep.size(); l-- > 0;) d = std::abs(a.g(i, int64_t(dep[l]))) + d;
        for (std::size_t l = ind.size(); l-- > 0;) e = std::abs(a.gi(i, int64_t(ind[l]))) + e;
        out[std::size_t(i)] = d + e;
    }
    return out;
}

/// A fresh identifier above those of the set and above `above`.
int64_t aux_fresh_id(const PolyZonotope &Z, int64_t above) {
    const auto &id = Z.ids();
    return std::max(id.empty() ? int64_t(0) : *std::max_element(id.begin(), id.end()), above) + 1;
}

/// Z with the columns of `cols` (n, k) as dependent generators on factors of their own, followed by
/// `zeros` zero columns on factors of their own (CORALean's absorbRounding: the error box of the
/// set made generators, which is empty at nearest). The new factors have the exponent 1 each.
PolyZonotope aux_make_dependent(const PolyZonotope &Z, const Tensor &cols, int64_t zeros,
                                int64_t above = 0) {
    const int64_t k = cols.shape()[1] + zeros, p = Z.num_factors();
    std::vector<int64_t> id = Z.ids();
    const int64_t fresh = aux_fresh_id(Z, above);
    for (int64_t j = 0; j < k; ++j) id.push_back(fresh + j);
    // the exponents: the old columns padded, a unit exponent for each new factor
    Exponents E;
    for (const std::vector<int> &col : Z.exponents()) {
        std::vector<int> c = col;
        c.resize(std::size_t(p + k), 0);
        E.push_back(std::move(c));
    }
    for (int64_t j = 0; j < k; ++j) {
        std::vector<int> c(std::size_t(p + k), 0);
        c[std::size_t(p + j)] = 1;
        E.push_back(std::move(c));
    }
    const Tensor none = Z.center().zeros_like({Z.dim(), 0});
    const Tensor padding = zeros > 0 ? Z.center().zeros_like({Z.dim(), zeros}) : none;
    return {Z.center(),    Tensor::concat({Z.generators(), cols, padding}, -1),
            none,          std::move(E),
            std::move(id), detail::Verbatim{}};
}

/// a * b, a NaN corner (0 * Inf) taken as 0.
double aux_mul(double a, double b) {
    const double p = a * b;
    return std::isnan(p) ? 0.0 : p;
}

/// The identifiers of Z2 shifted above those of Z1 (the factors of the two sets apart), and the
/// exponent matrices of the two sets over the identifiers of both.
struct Apart {
    std::vector<int64_t> id;
    Exponents E1, E2;
};

Apart aux_apart(const PolyZonotope &Z1, const PolyZonotope &Z2) {
    const int64_t shift =
        Z1.ids().empty() ? 0 : *std::max_element(Z1.ids().begin(), Z1.ids().end());
    Apart out;
    out.id = Z1.ids();
    for (const int64_t v : Z2.ids()) out.id.push_back(shift + v);
    // the exponents of both sets over the identifiers of both
    const std::size_t p1 = Z1.ids().size(), p2 = Z2.ids().size();
    for (const std::vector<int> &col : Z1.exponents()) {
        std::vector<int> c = col;
        c.resize(p1 + p2, 0);
        out.E1.push_back(std::move(c));
    }
    for (const std::vector<int> &col : Z2.exponents()) {
        std::vector<int> c(p1, 0);
        c.insert(c.end(), col.begin(), col.end());
        out.E2.push_back(std::move(c));
    }
    return out;
}

/// Which generators are boxed (the dependent ones, then the independent ones): the longest prefix
/// of the ranking of the metrics whose cumulative distance stays within dh_max, the rankings read
/// only as far as it goes.
std::vector<char> aux_boxed_prefix(const Host &a, const Vec &m_dep, const Vec &m_ind,
                                   double dh_max) {
    const auto n = static_cast<std::size_t>(a.n), h = m_dep.size();
    std::vector<char> boxed(h + m_ind.size(), 0);
    // both blocks ranked together are the merge of their rankings (a dependent first on a tie)
    LazyRank next_dep(m_dep), next_ind(m_ind);
    Vec acc_dep(n, 0.0), acc_ind(n, 0.0), v(n);
    double prev_dep = 0, prev_ind = 0, acc = 0;
    while (!next_dep.empty() || !next_ind.empty()) {
        const bool dependent = next_ind.empty() || (!next_dep.empty() &&
                                                    next_dep.peek().first <= next_ind.peek().first);
        double inc;
        std::size_t id;
        if (dependent) {
            // a dependent generator adds its magnitudes to the box
            id = next_dep.peek().second;
            next_dep.pop();
            for (std::size_t i = 0; i < n; ++i)
                acc_dep[i] = acc_dep[i] + std::abs(a.g(int64_t(i), int64_t(id)));
            const double dist = 0.5 * priv_norm_up(acc_dep);
            inc = dist - prev_dep;
            prev_dep = dist;
        } else {
            // an independent one adds its magnitudes, the first largest entry dropped
            const std::size_t j = next_ind.peek().second;
            next_ind.pop();
            id = h + j;
            std::size_t top = 0;
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = std::abs(a.gi(int64_t(i), int64_t(j)));
                if (v[top] < v[i]) top = i;
            }
            for (std::size_t i = 0; i < n; ++i) acc_ind[i] = acc_ind[i] + (i == top ? 0.0 : v[i]);
            const double dist = 2.0 * priv_norm_up(acc_ind);
            inc = dist - prev_ind;
            prev_ind = dist;
        }
        acc = acc + inc;
        if (!(acc <= dh_max)) break;
        boxed[id] = 1;
    }
    return boxed;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The reductions ----------------------------------------------------------------------------------

PolyZonotope priv_reduce_adaptive_poly(const PolyZonotope &Z, double diagpercent) {
    const Host a(Z);
    const int64_t n = a.n, h = a.h, q = a.q;
    const auto hh = static_cast<std::size_t>(h), qq = static_cast<std::size_t>(q);
    // the magnitudes of a dependent generator: even exponents only, at half
    std::vector<char> halfs(hh);
    for (int64_t j = 0; j < h; ++j) halfs[std::size_t(j)] = aux_all_even(a.E[std::size_t(j)]);
    const auto mag = [&](int64_t i, int64_t j) {
        return std::abs(halfs[std::size_t(j)] ? 0.5 * a.g(i, j) : a.g(i, j));
    };
    // the metrics: 1-norm of a dependent generator, 1-norm less the largest entry of an independent
    Vec m_dep(hh, 0.0), m_ind(qq, 0.0);
    for (int64_t i = n; i-- > 0;) {
        for (int64_t j = 0; j < h; ++j) m_dep[std::size_t(j)] = mag(i, j) + m_dep[std::size_t(j)];
        for (int64_t j = 0; j < q; ++j)
            m_ind[std::size_t(j)] = std::abs(a.gi(i, j)) + m_ind[std::size_t(j)];
    }
    for (int64_t j = 0; j < q; ++j) {
        double largest = std::abs(a.gi(0, j));
        for (int64_t i = 1; i < n; ++i) largest = std::max(largest, std::abs(a.gi(i, j)));
        m_ind[std::size_t(j)] = m_ind[std::size_t(j)] - largest;
    }
    // the allowed distance: the share of the norm of the box of the set
    Vec box(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        double d = 0, e = 0;
        for (int64_t j = h; j-- > 0;) d = mag(i, j) + d;
        for (int64_t j = q; j-- > 0;) e = std::abs(a.gi(i, j)) + e;
        box[std::size_t(i)] = d + e;
    }
    const double dh_max = (diagpercent * 0.5) * priv_norm_up(box);
    // the longest prefix of the ranking whose cumulative distance is within dHmax
    const std::vector<char> boxed = aux_boxed_prefix(a, m_dep, m_ind, dh_max);

    // the others stay, in their order; the boxed ones become the independent generators
    std::vector<std::size_t> keep, keep_i, drop, drop_i;
    for (int64_t j = 0; j < h; ++j) (boxed[std::size_t(j)] ? drop : keep).push_back(std::size_t(j));
    for (int64_t j = 0; j < q; ++j)
        (boxed[std::size_t(h + j)] ? drop_i : keep_i).push_back(std::size_t(j));
    return aux_compact(aux_reduce_by(Z, a, keep, keep_i, aux_drop_radius(a, drop, drop_i)));
}

// The dependent generators ------------------------------------------------------------------------

PolyZonotope priv_reduce_only_dep(const PolyZonotope &Z, int order) {
    const Host a(Z);
    if (a.h < int64_t(order + 1) * a.n) return Z;
    // the longest order * n stay, in the order of their length; the rest is boxed
    Vec length(std::size_t(a.h)), v(std::size_t(a.n));
    for (int64_t l = 0; l < a.h; ++l) {
        for (int64_t i = 0; i < a.n; ++i) v[std::size_t(i)] = a.g(i, l);
        length[std::size_t(l)] = priv_norm_up(v);
    }
    const std::vector<std::size_t> rank = aux_rank(length, true);
    const std::size_t kept =
        std::min<std::size_t>(std::size_t(order) * std::size_t(a.n), rank.size());
    const std::vector<std::size_t> keep(rank.begin(), rank.begin() + std::ptrdiff_t(kept)),
        drop(rank.begin() + std::ptrdiff_t(kept), rank.end());
    std::vector<std::size_t> keep_i(std::size_t(a.q));
    std::iota(keep_i.begin(), keep_i.end(), std::size_t(0));
    return aux_compact(aux_reduce_by(Z, a, keep, keep_i, aux_drop_radius(a, drop, {})));
}

// The sets of CORALean's float layer -------------------------------------------------------------

// The sums ----------------------------------------------------------------------------------------

PolyZonotope priv_poly_plus(const PolyZonotope &Z1, const PolyZonotope &Z2) {
    Apart a = aux_apart(Z1, Z2);
    Exponents E = std::move(a.E1);
    E.insert(E.end(), a.E2.begin(), a.E2.end());
    return {Z1.center() + Z2.center(),
            Tensor::concat({Z1.generators(), Z2.generators()}, -1),
            Tensor::concat({Z1.independent_generators(), Z2.independent_generators()}, -1),
            std::move(E),
            std::move(a.id),
            detail::Verbatim{}};
}

PolyZonotope priv_poly_plus(const PolyZonotope &Z, const Zonotope &Z2) {
    return {Z.center() + Z2.center(),
            Z.generators(),
            Tensor::concat({Z.independent_generators(), Z2.generators()}, -1),
            Z.exponents(),
            Z.ids(),
            detail::Verbatim{}};
}

PolyZonotope priv_poly_exact_plus(const PolyZonotope &Z1, const PolyZonotope &Z2) {
    // the factors of both over the union of the identifiers: the shorter list is padded
    std::vector<int64_t> id = Z1.ids();
    id.insert(id.end(), Z2.ids().begin(), Z2.ids().end());
    std::sort(id.begin(), id.end());
    id.erase(std::unique(id.begin(), id.end()), id.end());
    // the exponents of a set over the union of the identifiers
    const auto over = [&](const PolyZonotope &Z) {
        std::vector<std::size_t> pos(Z.ids().size());
        for (std::size_t k = 0; k < pos.size(); ++k)
            pos[k] = std::size_t(std::lower_bound(id.begin(), id.end(), Z.ids()[k]) - id.begin());
        Exponents E;
        for (const std::vector<int> &col : Z.exponents()) {
            std::vector<int> c(id.size(), 0);
            for (std::size_t k = 0; k < pos.size(); ++k) c[pos[k]] = col[k];
            E.push_back(std::move(c));
        }
        return E;
    };
    // the exponents of both sets side by side, the columns of Z1 first
    Exponents E = over(Z1);
    const Exponents E2 = over(Z2);
    E.insert(E.end(), E2.begin(), E2.end());
    return {Z1.center() + Z2.center(),
            Tensor::concat({Z1.generators(), Z2.generators()}, -1),
            Tensor::concat({Z1.independent_generators(), Z2.independent_generators()}, -1),
            std::move(E),
            std::move(id),
            detail::Verbatim{}};
}

// The hull of two sets ----------------------------------------------------------------------------

PolyZonotope priv_poly_lin_comb(const PolyZonotope &Z1, const PolyZonotope &Z2) {
    const Host a(Z1), b(Z2);
    const int64_t n = a.n, h1 = a.h, h2 = b.h;
    const Tensor &like = Z1.center();
    // the columns [G1/2, G2/2 | G1/2, -G2/2, (c1 - c2)/2] and the independent block of the pairs
    const int64_t h = 2 * (h1 + h2) + 1;
    Vec G(std::size_t(n * h));
    Vec c(static_cast<std::size_t>(n));
    // the columns of the pairs, the sliding generator and the center
    for (int64_t i = 0; i < n; ++i) {
        c[std::size_t(i)] = (a.c[std::size_t(i)] + b.c[std::size_t(i)]) * 0.5;
        int64_t col = 0;
        const auto put = [&](double v) { G[std::size_t(i * h + col++)] = v; };
        for (int64_t l = 0; l < h1; ++l) put(a.g(i, l) * 0.5);
        for (int64_t l = 0; l < h2; ++l) put(b.g(i, l) * 0.5);
        for (int64_t l = 0; l < h1; ++l) put(a.g(i, l) * 0.5);
        for (int64_t l = 0; l < h2; ++l) put(-(b.g(i, l) * 0.5));
        put((a.c[std::size_t(i)] - b.c[std::size_t(i)]) * 0.5);
    }
    const int64_t q = a.q + b.q;
    Vec GI(std::size_t(n * q));
    for (int64_t i = 0; i < n; ++i) {
        int64_t col = 0;
        const auto put = [&](double v) { GI[std::size_t(i * q + col++)] = v; };
        for (int64_t l = 0; l < a.q; ++l)
            put(l < b.q ? (a.gi(i, l) + b.gi(i, l)) * 0.5 : a.gi(i, l));
        for (int64_t l = 0; l < b.q; ++l)
            put(l < a.q ? (a.gi(i, l) - b.gi(i, l)) * 0.5 : b.gi(i, l));
    }
    // the exponents: both blocks over the factors of both sets, the fresh factor 1 on the second
    Apart ap = aux_apart(Z1, Z2);
    const std::size_t p = ap.id.size();
    Exponents all = ap.E1;
    all.insert(all.end(), ap.E2.begin(), ap.E2.end());
    // both blocks over the factors of both sets, the fresh factor 1 on the second
    Exponents E;
    for (int block = 0; block < 2; ++block) {
        for (const std::vector<int> &col : all) {
            std::vector<int> e = col;
            e.push_back(block);
            E.push_back(std::move(e));
        }
    }
    std::vector<int> last(p + 1, 0);
    last.back() = 1;
    E.push_back(std::move(last));
    std::vector<int64_t> id = ap.id;
    id.push_back((id.empty() ? 0 : *std::max_element(id.begin(), id.end())) + 1);
    return {Tensor::like(like, c, {n, 1}),
            aux_tensor(like, G, n, h),
            aux_tensor(like, GI, n, q),
            std::move(E),
            std::move(id),
            detail::Verbatim{}};
}

// The map by an interval matrix -------------------------------------------------------------------

PolyZonotope priv_poly_interval_map(const Interval &M, const PolyZonotope &Z) {
    const Host a(Z);
    const Tensor &like = Z.center();
    const auto m = M.inf().size(-2), n = a.n;
    const Vec lo = M.inf().data(), hi = M.sup().data();
    // mid(M) and the box M - mid(M)
    Vec C(std::size_t(m * n)), Rlo(C.size()), Rhi(C.size());
    for (std::size_t e = 0; e < C.size(); ++e) {
        C[e] = (lo[e] + hi[e]) * 0.5;
        Rlo[e] = lo[e] - C[e];
        Rhi[e] = hi[e] - C[e];
    }
    const auto dot = [&](int64_t i, const auto &column) {
        double sum = 0;
        for (int64_t k = n; k-- > 0;) sum = aux_mul(C[std::size_t(i * n + k)], column(k)) + sum;
        return sum;
    };
    Vec c(static_cast<std::size_t>(m)), G(static_cast<std::size_t>(m * a.h)),
        GI(static_cast<std::size_t>(m * (a.q + m)), 0.0);
    for (int64_t i = 0; i < m; ++i) {
        c[std::size_t(i)] = dot(i, [&](int64_t k) { return a.c[std::size_t(k)]; });
        for (int64_t l = 0; l < a.h; ++l)
            G[std::size_t(i * a.h + l)] = dot(i, [&](int64_t k) { return a.g(k, l); });
        for (int64_t l = 0; l < a.q; ++l)
            GI[std::size_t(i * (a.q + m) + l)] = dot(i, [&](int64_t k) { return a.gi(k, l); });
    }
    // the free box of Z: c -+ the row sums of both blocks, subtracted in turn
    Vec blo(static_cast<std::size_t>(n)), bhi(static_cast<std::size_t>(n));
    for (int64_t k = 0; k < n; ++k) {
        double rg = 0, rgi = 0;
        for (int64_t l = a.h; l-- > 0;) rg = std::abs(a.g(k, l)) + rg;
        for (int64_t l = a.q; l-- > 0;) rgi = std::abs(a.gi(k, l)) + rgi;
        blo[std::size_t(k)] = (a.c[std::size_t(k)] - rg) - rgi;
        bhi[std::size_t(k)] = (a.c[std::size_t(k)] + rg) + rgi;
    }
    // [M - mid(M)] times that box, as a zonotope of its stored center and a diagonal of radii
    for (int64_t i = 0; i < m; ++i) {
        // each coordinate: the sums of the smallest and largest corner products
        double s_lo = 0, s_hi = 0;
        for (int64_t k = n; k-- > 0;) {
            const std::size_t e = std::size_t(i * n + k);
            const double p[4] = {
                aux_mul(Rlo[e], blo[std::size_t(k)]), aux_mul(Rlo[e], bhi[std::size_t(k)]),
                aux_mul(Rhi[e], blo[std::size_t(k)]), aux_mul(Rhi[e], bhi[std::size_t(k)])};
            s_lo = std::min(std::min(p[0], p[1]), std::min(p[2], p[3])) + s_lo;
            s_hi = std::max(std::max(p[0], p[1]), std::max(p[2], p[3])) + s_hi;
        }
        const double center = (s_lo + s_hi) * 0.5;
        c[std::size_t(i)] = c[std::size_t(i)] + center;
        GI[std::size_t(i * (a.q + m) + a.q + i)] = std::max(s_hi - center, center - s_lo);
    }
    return {Tensor::like(like, c, {m, 1}),
            aux_tensor(like, G, m, a.h),
            aux_tensor(like, GI, m, a.q + m),
            Z.exponents(),
            Z.ids(),
            detail::Verbatim{}};
}

// The restructuring -------------------------------------------------------------------------------

PolyZonotope priv_restructure(const PolyZonotope &Z) {
    const Host a(Z);
    // the independent part boxed into one diagonal generator per coordinate, made dependent
    std::vector<std::size_t> all(std::size_t(a.h)), none, ind(std::size_t(a.q));
    std::iota(all.begin(), all.end(), std::size_t(0));
    std::iota(ind.begin(), ind.end(), std::size_t(0));
    const Vec radius = aux_drop_radius(a, none, ind);
    const Tensor diag = Tensor::like(Z.center(), radius, {a.n, 1}).diag();
    return aux_make_dependent(Z, diag, a.n);
}

PolyZonotope priv_absorb_independent(const PolyZonotope &Z, int64_t above) {
    return aux_make_dependent(Z, Z.independent_generators(), Z.dim(), above);
}

// The volume --------------------------------------------------------------------------------------

Vec priv_poly_radius(const PolyZonotope &Z) {
    const Interval box = Z.zonotope().interval();
    const Vec lo = box.inf().data(), hi = box.sup().data();
    Vec out(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) out[i] = 0.5 * (hi[i] - lo[i]);
    return out;
}

bool priv_volume_ratio_exceeds(const PolyZonotope &Z, double ratio) {
    const Host a(Z);
    if (a.q == 0) return false; // no independent generators: the ratio is 0
    if (a.h == 0) return true;  // no dependent ones: the ratio is Inf
    Vec rad_ind(std::size_t(a.n));
    for (int64_t i = 0; i < a.n; ++i) {
        Vec v(std::size_t(a.q));
        for (int64_t j = 0; j < a.q; ++j) v[std::size_t(j)] = std::abs(a.gi(i, j));
        rad_ind[std::size_t(i)] = priv_sum_up(v);
    }
    const PolyZonotope dep(Z.center(), Z.generators(), Z.center().zeros_like({a.n, 0}),
                           Z.exponents(), Z.ids());
    const Vec rad_dep = priv_poly_radius(dep);
    // the dimensions where both parts have width
    double vol_ind = 1, vol_dep = 1;
    for (std::size_t i = 0; i < rad_ind.size(); ++i)
        if (rad_dep[i] > 0 && rad_ind[i] > 0) {
            vol_ind = vol_ind * (rad_ind[i] + rad_ind[i]);
            vol_dep = vol_dep * (rad_dep[i] + rad_dep[i]);
        }
    return !(vol_ind <= priv_pow(ratio, int(a.n)) * vol_dep);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
