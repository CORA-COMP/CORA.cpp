// reduce_adaptive - CORA's zonotope.reduce with method "adaptive", as CORALean's reduceAdaptive
//
// The generators are ranked by Girard's metric ||g||_1 - ||g||_inf (zero on an axis-aligned one,
// ties in index order). Those with metric 0 are boxed, then as many more as keep twice the norm of
// the box of the cumulated entries (each generator without its largest entry) within twice
// diag_percent times the norm of the box of Z. The others stay in their order; the result has
// them, then one diagonal generator per dimension for the box. Sums add from the last term to the
// first, as CORALean's sumUp. The cumulated box only grows, so the ranking is produced
// (lazy_rank.h) only as far as the bound stays within the allowed distance.
//
// Syntax:   Zred = Z.reduce_adaptive(diag_percent);
// Inputs:   diag_percent - the share of the box's norm the Hausdorff distance may reach
// Outputs:  Zred - a zonotope that encloses Z
// See also: reduce
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceAdaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/lazy_rank.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Sums and norms ----------------------------------------------------------------------------------

/// The Euclidean norm of the magnitudes, rounded to nearest (sqrt of the sum of squares). Each
/// square is rounded before it is added (this file is compiled without fused multiply-add).
double aux_norm(const std::vector<double> &v) {
    double sum = 0;
    for (std::size_t i = v.size(); i-- > 0;) sum = std::abs(v[i]) * std::abs(v[i]) + sum;
    return std::sqrt(sum);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The reduction -----------------------------------------------------------------------------------

Zonotope Zonotope::reduce_adaptive(double diag_percent) const {
    detail::Entry entry;
    CORA_CHECK(diag_percent >= 0 && diag_percent < 1e300, Error::wrong_value, [&] {
        return "reduce_adaptive: diag_percent must be nonnegative and finite, got " +
               std::to_string(diag_percent);
    });
    CORA_CHECK(c_.ndim() == 2 && G_.ndim() == 2, Error::not_supported, [] {
        return std::string("reduce_adaptive: a batch of zonotopes is not supported");
    });
    const int64_t n = dim(), h = G_.shape().back();
    const std::vector<double> g = G_.data();
    const auto a = [&](std::size_t at) { return std::abs(g[at]); };

    // the box (row sums) and Girard's metric (column sums less the largest entry) in one pass
    std::vector<double> box(static_cast<std::size_t>(n)), metric(static_cast<std::size_t>(h), 0.0),
        largest(static_cast<std::size_t>(h), 0.0);
    for (int64_t i = n; i-- > 0;) {
        double sum = 0;
        for (int64_t j = h; j-- > 0;) {
            const double x = a(std::size_t(i * h + j));
            sum = x + sum;
            metric[std::size_t(j)] = x + metric[std::size_t(j)];
            largest[std::size_t(j)] = std::max(largest[std::size_t(j)], x);
        }
        box[std::size_t(i)] = sum;
    }
    const double d_h_max = (diag_percent * 2) * aux_norm(box);

    // the generators with metric 0 are boxed, the others are ranked (stable, smallest first)
    std::vector<char> boxed(static_cast<std::size_t>(h), 0);
    std::vector<LazyRank::Entry> rest;
    rest.reserve(static_cast<std::size_t>(h));
    for (int64_t j = 0; j < h; ++j) {
        const double m = metric[std::size_t(j)] - largest[std::size_t(j)];
        if (m <= 0)
            boxed[std::size_t(j)] = 1;
        else
            rest.emplace_back(m, std::size_t(j));
    }

    // the longest prefix of the rest within the allowed distance (the bound only grows)
    LazyRank rank(std::move(rest));
    std::vector<double> cumulated(static_cast<std::size_t>(n), 0.0),
        col(static_cast<std::size_t>(n));
    while (!rank.empty()) {
        const std::size_t j = rank.peek().second;
        // the first of the largest entries drops out
        std::size_t top = 0;
        for (std::size_t i = 0; i < col.size(); ++i) {
            col[i] = a(i * std::size_t(h) + j);
            if (col[top] < col[i]) top = i;
        }
        for (std::size_t i = 0; i < col.size(); ++i)
            if (i != top) cumulated[i] = cumulated[i] + col[i];
        if (!(2 * aux_norm(cumulated) <= d_h_max)) break;
        boxed[j] = 1;
        rank.pop();
    }

    // the kept generators in index order, then the box of the reduced ones
    std::vector<int64_t> kept;
    kept.reserve(static_cast<std::size_t>(h));
    for (int64_t j = 0; j < h; ++j)
        if (!boxed[std::size_t(j)]) kept.push_back(j);
    const int64_t q = static_cast<int64_t>(kept.size());
    std::vector<double> out(static_cast<std::size_t>(n * (q + n)), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t l = 0; l < q; ++l)
            out[std::size_t(i * (q + n) + l)] = g[std::size_t(i * h + kept[std::size_t(l)])];
        double sum = 0;
        for (int64_t j = h; j-- > 0;)
            if (boxed[std::size_t(j)]) sum = a(std::size_t(i * h + j)) + sum;
        out[std::size_t(i * (q + n) + q + i)] = sum;
    }
    return {c_, Tensor::like(c_, out, {n, q + n})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
