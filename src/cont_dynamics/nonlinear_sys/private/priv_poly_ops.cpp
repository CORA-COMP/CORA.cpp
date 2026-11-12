// priv_poly_ops - reductions of polynomial zonotopes for poly-adaptive
//
// The functions are CORALean's float definitions of CORA's reduce(pZ, 'adaptive', diagpercent)
// (the generators ranked by their length, the longest prefix whose box distance stays within
// diagpercent boxed, an even-exponent generator at its full radius), aux_reduceOnlyDep (past
// (order + 1) n dependent generators the shortest are boxed), restructure (the independent part
// boxed and made dependent) and approxVolumeRatio. Sums add from the last term to the first and a
// norm is the root of the sum of squares, as CORALean's sumUp and normUp at nearest.
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
    Exponents E;

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

/// The indices 0 .. m-1 sorted by the key, the smaller first, ties in index order.
std::vector<std::size_t> aux_rank(const Vec &key, bool descending = false) {
    std::vector<std::size_t> idx(key.size());
    std::iota(idx.begin(), idx.end(), std::size_t(0));
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return descending ? key[b] < key[a] : key[a] < key[b];
    });
    return idx;
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
    Exponents E;
    for (const std::size_t l : keep) E.push_back(a.E[l]);
    return {Z.center(), aux_tensor(like, G, n, hk), aux_tensor(like, GI, n, qk + n), std::move(E),
            Z.ids(),    detail::Normalized{}};
}

/// The radius per coordinate of the dependent generators `dep` and the independent ones `ind`: the
/// row sums of the magnitudes, added upward (CORALean's dropRadius).
Vec aux_drop_radius(const Host &a, const std::vector<std::size_t> &dep,
                    const std::vector<std::size_t> &ind) {
    Vec out(std::size_t(a.n));
    for (int64_t i = 0; i < a.n; ++i) {
        Vec d, e;
        for (const std::size_t l : dep) d.push_back(std::abs(a.g(i, int64_t(l))));
        for (const std::size_t l : ind) e.push_back(std::abs(a.gi(i, int64_t(l))));
        out[std::size_t(i)] = priv_sum_up(d) + priv_sum_up(e);
    }
    return out;
}

/// A fresh identifier above those of the set and above `above`.
int64_t aux_fresh_id(const PolyZonotope &Z, int64_t above) {
    const auto &id = Z.ids();
    return std::max(id.empty() ? int64_t(0) : *std::max_element(id.begin(), id.end()), above) + 1;
}

/// Z with the columns of `cols` (n, k) as dependent generators on factors of their own.
PolyZonotope aux_make_dependent(const PolyZonotope &Z, const Tensor &cols, const Tensor &GI,
                                int64_t above = 0) {
    const int64_t k = cols.shape()[1], p = Z.num_factors();
    std::vector<int64_t> id = Z.ids();
    const int64_t fresh = aux_fresh_id(Z, above);
    for (int64_t j = 0; j < k; ++j) id.push_back(fresh + j);
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
    return {Z.center(), Tensor::concat({Z.generators(), cols}, -1), GI, std::move(E),
            std::move(id)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The reductions ----------------------------------------------------------------------------------

PolyZonotope priv_reduce_adaptive_poly(const PolyZonotope &Z, double diagpercent) {
    const Host a(Z);
    const int64_t n = a.n, h = a.h, q = a.q;
    const auto nn = static_cast<std::size_t>(n), hh = static_cast<std::size_t>(h),
               qq = static_cast<std::size_t>(q);
    if (h + q == 0) return Z;
    // the metrics: 1-norm of a dependent generator, 1-norm less the largest entry of an independent
    std::vector<char> halfs(hh);
    for (int64_t j = 0; j < h; ++j) halfs[std::size_t(j)] = aux_all_even(a.E[std::size_t(j)]);
    const auto column = [&](int64_t j) {
        Vec v(nn);
        for (int64_t i = 0; i < n; ++i)
            v[std::size_t(i)] = std::abs(halfs[std::size_t(j)] ? 0.5 * a.g(i, j) : a.g(i, j));
        return v;
    };
    // the magnitudes of an independent generator
    const auto column_i = [&](int64_t j) {
        Vec v(nn);
        for (int64_t i = 0; i < n; ++i) v[std::size_t(i)] = std::abs(a.gi(i, j));
        return v;
    };
    // the metrics of all generators, ranked together (stable, smallest first)
    Vec m_dep(hh), m_ind(qq);
    for (int64_t j = 0; j < h; ++j) m_dep[std::size_t(j)] = priv_sum_up(column(j));
    for (int64_t j = 0; j < q; ++j) {
        const Vec v = column_i(j);
        m_ind[std::size_t(j)] = priv_sum_up(v) - *std::max_element(v.begin(), v.end());
    }
    Vec metric = m_dep;
    metric.insert(metric.end(), m_ind.begin(), m_ind.end());
    const std::vector<std::size_t> rank = aux_rank(metric);

    // the increments of the box distance in the ranking of each block (dependent first)
    Vec inc(hh + qq, 0.0);
    {
        Vec acc(nn, 0.0);
        double previous = 0;
        for (const std::size_t j : aux_rank(m_dep)) {
            for (int64_t i = 0; i < n; ++i)
                acc[std::size_t(i)] = acc[std::size_t(i)] + std::abs(a.g(i, int64_t(j)));
            const double dist = 0.5 * priv_norm_up(acc);
            inc[j] = dist - previous;
            previous = dist;
        }
    }
    // the independent ones: the largest entry of each generator drops out
    {
        Vec acc(nn, 0.0);
        double previous = 0;
        for (const std::size_t j : aux_rank(m_ind)) {
            const Vec v = column_i(int64_t(j));
            const std::size_t top = std::size_t(
                std::find(v.begin(), v.end(), *std::max_element(v.begin(), v.end())) - v.begin());
            for (std::size_t i = 0; i < std::size_t(n); ++i)
                acc[i] = acc[i] + (i == top ? 0.0 : v[i]);
            const double dist = 2.0 * priv_norm_up(acc);
            inc[std::size_t(h) + j] = dist - previous;
            previous = dist;
        }
    }

    // the longest prefix of the ranking whose cumulative distance is within dHmax
    Vec box(nn);
    for (int64_t i = 0; i < n; ++i) {
        Vec d(hh), e(qq);
        for (int64_t j = 0; j < h; ++j)
            d[std::size_t(j)] = std::abs(halfs[std::size_t(j)] ? 0.5 * a.g(i, j) : a.g(i, j));
        for (int64_t j = 0; j < q; ++j) e[std::size_t(j)] = std::abs(a.gi(i, j));
        box[std::size_t(i)] = priv_sum_up(d) + priv_sum_up(e);
    }
    const double dh_max = (diagpercent * 0.5) * priv_norm_up(box);
    double acc = 0;
    std::size_t count = 0, until = 0;
    for (const std::size_t j : rank) {
        acc = acc + inc[j];
        ++count;
        if (acc <= dh_max) until = count;
    }
    std::vector<char> boxed(hh + qq, 0);
    for (std::size_t k = 0; k < until; ++k) boxed[rank[k]] = 1;

    // the others stay, in their order; the boxed ones become the independent generators
    std::vector<std::size_t> keep, keep_i, drop, drop_i;
    for (int64_t j = 0; j < h; ++j) (boxed[std::size_t(j)] ? drop : keep).push_back(std::size_t(j));
    for (int64_t j = 0; j < q; ++j)
        (boxed[std::size_t(h + j)] ? drop_i : keep_i).push_back(std::size_t(j));
    return aux_reduce_by(Z, a, keep, keep_i, aux_drop_radius(a, drop, drop_i)).compact();
}

// The dependent generators ------------------------------------------------------------------------

PolyZonotope priv_reduce_only_dep(const PolyZonotope &Z, int order) {
    const Host a(Z);
    if (a.h < int64_t(order + 1) * a.n) return Z;
    // the longest order * n stay, in the order of their length; the rest is boxed
    Vec length(std::size_t(a.h));
    for (int64_t l = 0; l < a.h; ++l) {
        Vec v(std::size_t(a.n));
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
    return aux_reduce_by(Z, a, keep, keep_i, aux_drop_radius(a, drop, {})).compact();
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
    return aux_make_dependent(Z, diag, Z.center().zeros_like({a.n, 0}));
}

PolyZonotope priv_absorb_independent(const PolyZonotope &Z, int64_t above) {
    if (Z.independent_generators().shape()[1] == 0) return Z;
    return aux_make_dependent(Z, Z.independent_generators(), Z.center().zeros_like({Z.dim(), 0}),
                              above);
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
