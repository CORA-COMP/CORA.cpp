// reduce_adaptive - Girard's reduction with as many generators boxed as a Hausdorff bound allows
//
// The generators are sorted by Girard's metric ||g||_1 - ||g||_inf; those with metric 0 (axis
// aligned) are boxed for free. Of the rest, the shortest are boxed one after the other while the
// Hausdorff distance of the reduction to the original, bounded by the norm of the cumulated
// generators without their largest entry (twice), stays below diag_percent times twice the norm of
// the radius of the box of the zonotope. The boxed generators become n axis generators (also the
// zero ones, as in CORA). Without generators, or if every generator is axis aligned, the result
// is the box.
//
// Syntax:   AdaptiveReduction r = Z.reduce_adaptive(diag_percent);   // r.Z, r.error, r.idx
// Inputs:   diag_percent - the allowed Hausdorff distance as a share of the box diagonal, >= 0
// Outputs:  r - the reduced zonotope, its error bound and the 0-based indices of the boxed ones
// See also: reduce, reduce_idx
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceAdaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
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

// The order of the generators ---------------------------------------------------------------------

/// The indices of the m generators (the absolute values a, n x m row-major) sorted by Girard's
/// metric ||g||_1 - ||g||_inf, ascending and stable, and the number with metric 0.
std::pair<std::vector<int64_t>, int64_t> aux_girard_order(const std::vector<double> &a, int64_t n,
                                                          int64_t m) {
    std::vector<double> metric(static_cast<std::size_t>(m));
    for (int64_t j = 0; j < m; ++j) {
        // the sum of the entries of the generator minus the largest one
        double sum = 0, top = 0;
        for (int64_t i = 0; i < n; ++i) {
            const double v = a[std::size_t(i * m + j)];
            sum += v;
            top = std::max(top, v);
        }
        metric[std::size_t(j)] = sum - top;
    }
    std::vector<int64_t> idx(static_cast<std::size_t>(m));
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t p, int64_t q) {
        return metric[std::size_t(p)] < metric[std::size_t(q)];
    });
    return {idx, int64_t(std::count(metric.begin(), metric.end(), 0.0))};
}

// The bound of the Hausdorff distance -------------------------------------------------------------

/// For each prefix of the generators idx[last0:], boxed together, twice the norm of the sum of
/// their entries without the largest entry of each generator (the bound on the distance).
std::vector<double> aux_prefix_bounds(const std::vector<double> &a, int64_t n, int64_t m,
                                      const std::vector<int64_t> &idx, int64_t last0) {
    std::vector<double> cumulated(std::size_t(n), 0.0), bounds;
    for (int64_t col = last0; col < m; ++col) {
        const auto src = std::size_t(idx[std::size_t(col)]);
        // the largest entry of the generator drops out (the first one on a tie)
        std::size_t arg = 0;
        for (std::size_t i = 1; i < std::size_t(n); ++i)
            if (a[i * std::size_t(m) + src] > a[arg * std::size_t(m) + src]) arg = i;
        double norm_sq = 0;
        for (std::size_t i = 0; i < std::size_t(n); ++i) {
            if (i != arg) cumulated[i] += a[i * std::size_t(m) + src];
            norm_sq += cumulated[i] * cumulated[i];
        }
        bounds.push_back(2 * std::sqrt(norm_sq));
    }
    return bounds;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The reduction -----------------------------------------------------------------------------------

AdaptiveReduction Zonotope::reduce_adaptive(double diag_percent) const {
    detail::Entry entry;
    CORA_CHECK(diag_percent >= 0 && diag_percent < 1e300, Error::wrong_value, [&] {
        return "reduce_adaptive: diag_percent must be nonnegative and finite, got " +
               std::to_string(diag_percent);
    });
    CORA_CHECK(c_.ndim() == 2 && G_.ndim() == 2, Error::not_supported, [] {
        return std::string("reduce_adaptive: a batch of zonotopes is not supported");
    });
    const int64_t n = dim(), m = G_.shape().back();
    if (m == 0) return {*this, 0.0, {}};

    // the box of the generators, the allowed distance, and the generators by Girard's metric
    const Tensor Gabs = G_.abs();
    const Tensor Gbox = Gabs.sum(-1, true);
    const std::vector<double> box = Gbox.data(), a = Gabs.data();
    double box_sq = 0;
    for (const double b : box) box_sq += b * b;
    const double d_h_max = (diag_percent * 2) * std::sqrt(box_sq);
    const auto [idx, last0] = aux_girard_order(a, n, m);

    // every generator axis aligned: the box
    if (last0 == m) {
        std::vector<int64_t> keep;
        for (int64_t i = 0; i < n; ++i)
            if (box[std::size_t(i)] != 0) keep.push_back(i);
        return {Zonotope(c_, Gbox.diag().take(keep, -1)), 0.0, idx};
    }

    // the last prefix of the rest whose bound is within the allowed distance
    const std::vector<double> h = aux_prefix_bounds(a, n, m, idx, last0);
    int64_t red = 0;
    for (int64_t col = m - last0; col >= 1; --col)
        if (h[std::size_t(col - 1)] <= d_h_max) {
            red = col;
            break;
        }
    const double error = red > 0 ? h[std::size_t(red - 1)] : 0.0;
    const std::vector<int64_t> boxed(idx.begin(), idx.begin() + last0 + red);
    std::vector<int64_t> kept(idx.begin() + last0 + red, idx.end());
    std::sort(kept.begin(), kept.end());

    // the kept generators and the box of the boxed ones
    const std::vector<int64_t> zero_cols(idx.begin(), idx.begin() + last0);
    const std::vector<int64_t> red_cols(idx.begin() + last0, idx.begin() + last0 + red);
    const Tensor G_zero = last0 > 0 ? Gabs.take(zero_cols, -1).sum(-1, true) : c_.zeros_like();
    const Tensor G_red = red > 0 ? Gabs.take(red_cols, -1).sum(-1, true) : c_.zeros_like();
    const Tensor diag = (G_red + G_zero).diag();
    if (kept.empty()) return {Zonotope(c_, diag), error, boxed};
    return {Zonotope(c_, Tensor::concat({G_.take(kept, -1), diag}, -1)), error, boxed};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
