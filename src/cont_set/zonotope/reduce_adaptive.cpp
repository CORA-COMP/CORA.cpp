// reduce_adaptive - CORA's zonotope.reduce with method "adaptive", as CORALean's reduceAdaptive
//
// The generators are ranked by Girard's metric ||g||_1 - ||g||_inf (zero on an axis-aligned one,
// ties in index order). Those with metric 0 are boxed, then as many more as keep twice the norm of
// the box of the cumulated entries (each generator without its largest entry) within twice
// diag_percent times the norm of the box of Z. The others stay in their order; the result has
// them, then one diagonal generator per dimension for the box. Sums add from the last term to the
// first, as CORALean's sumUp.
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

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Sums and norms ----------------------------------------------------------------------------------

/// The sum of v from the last entry to the first (CORALean's sumUp).
double aux_sum_up(const std::vector<double> &v) {
    double sum = 0;
    for (std::size_t i = v.size(); i-- > 0;) sum = v[i] + sum;
    return sum;
}

/// The Euclidean norm of the magnitudes, rounded to nearest (sqrt of the sum of squares).
double aux_norm(const std::vector<double> &v) {
    std::vector<double> squares(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) squares[i] = std::abs(v[i]) * std::abs(v[i]);
    return std::sqrt(aux_sum_up(squares));
}

/// The absolute entries of generator j (a is n x h row-major).
std::vector<double> aux_column(const std::vector<double> &a, int64_t n, int64_t h, int64_t j) {
    std::vector<double> col(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) col[std::size_t(i)] = a[std::size_t(i * h + j)];
    return col;
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
    const std::vector<double> g = G_.data(), a = G_.abs().data();

    // the box of the generators and the allowed distance
    std::vector<double> box(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        std::vector<double> row(static_cast<std::size_t>(h));
        for (int64_t j = 0; j < h; ++j) row[std::size_t(j)] = a[std::size_t(i * h + j)];
        box[std::size_t(i)] = aux_sum_up(row);
    }
    const double d_h_max = (diag_percent * 2) * aux_norm(box);

    // Girard's metric of each generator, and the ranking (stable, smallest first)
    std::vector<double> metric(static_cast<std::size_t>(h));
    for (int64_t j = 0; j < h; ++j) {
        const std::vector<double> col = aux_column(a, n, h, j);
        metric[std::size_t(j)] = aux_sum_up(col) - *std::max_element(col.begin(), col.end());
    }
    std::vector<int64_t> ranked(static_cast<std::size_t>(h));
    std::iota(ranked.begin(), ranked.end(), int64_t{0});
    std::stable_sort(ranked.begin(), ranked.end(), [&](int64_t p, int64_t q) {
        return metric[std::size_t(p)] < metric[std::size_t(q)];
    });
    int64_t zeros = 0;
    for (const int64_t j : ranked) zeros += metric[std::size_t(j)] <= 0;

    // the longest prefix of the rest, boxed, whose bound stays within the allowed distance
    std::vector<double> cumulated(static_cast<std::size_t>(n), 0.0);
    int64_t count = 0, last = 0;
    for (int64_t k = zeros; k < h; ++k) {
        const std::vector<double> col = aux_column(a, n, h, ranked[std::size_t(k)]);
        // the first of the largest entries drops out
        const auto top = std::size_t(std::max_element(col.begin(), col.end()) - col.begin());
        for (std::size_t i = 0; i < col.size(); ++i)
            if (i != top) cumulated[i] = cumulated[i] + col[i];
        ++count;
        if (2 * aux_norm(cumulated) <= d_h_max) last = count;
    }
    const int64_t reduced = std::min(zeros + last, h);

    // the kept generators in index order, then the box of the reduced ones
    std::vector<char> boxed(static_cast<std::size_t>(h), 0);
    for (int64_t k = 0; k < reduced; ++k) boxed[std::size_t(ranked[std::size_t(k)])] = 1;
    std::vector<int64_t> kept;
    for (int64_t j = 0; j < h; ++j)
        if (!boxed[std::size_t(j)]) kept.push_back(j);
    const int64_t q = static_cast<int64_t>(kept.size());
    std::vector<double> out(static_cast<std::size_t>(n * (q + n)), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t l = 0; l < q; ++l)
            out[std::size_t(i * (q + n) + l)] = g[std::size_t(i * h + kept[std::size_t(l)])];
        std::vector<double> row;
        for (int64_t j = 0; j < h; ++j)
            if (boxed[std::size_t(j)]) row.push_back(a[std::size_t(i * h + j)]);
        out[std::size_t(i * (q + n) + q + i)] = aux_sum_up(row);
    }
    return {c_, Tensor::like(c_, out, {n, q + n})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
