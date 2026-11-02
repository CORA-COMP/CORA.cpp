// reduce - fewer generators for a zonotope, as CORA's zonotope.reduce with method "girard"
//
// The generators that cost the most to box stay, by Girard's metric ||g||_1 - ||g||_inf (zero on an
// axis-aligned one); the others are replaced by the box that encloses them. The result contains the
// zonotope and has at most order * n generators (n the dimension).
//
// Syntax:   Zred = Z.reduce(order);
// Inputs:   order - the largest number of generators per dimension, at least 1
// Outputs:  Zred - a zonotope that encloses Z; Z itself if it has few enough generators
// See also: interval, plus
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceGirard.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"
#include "cora/global/error.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the zonotope order must be at least 1, got " + std::to_string(order);
    });
    const std::vector<int64_t> shape = G_.shape();
    CORA_CHECK(shape.size() == 2, Error::not_supported, [&] {
        return "reduce: takes a single zonotope, not a batch of the shape " + shape_text(shape) +
               "; reduce the members one by one";
    });
    const int64_t n = shape[0], m = shape[1];
    if (m <= order * n) return *this;

    // Which generators stay is decided on the metric; the sets are then built from tensors.
    const Tensor abs_t = G_.abs().transpose();
    const std::vector<double> metric = (abs_t.sum_last() - abs_t.max_last()).data();
    std::vector<int64_t> by_metric(m);
    std::iota(by_metric.begin(), by_metric.end(), 0);
    std::stable_sort(by_metric.begin(), by_metric.end(),
                     [&](int64_t a, int64_t b) { return metric[a] > metric[b]; });

    // The kept generators come first, then one box generator per dimension.
    const int64_t keep = order * n - n;
    const std::vector<int64_t> kept(by_metric.begin(), by_metric.begin() + keep);
    const std::vector<int64_t> rest(by_metric.begin() + keep, by_metric.end());
    const Tensor box = G_.select_cols(rest).abs().sum_last().diag();
    return {c_, keep == 0 ? box : Tensor::cat_last({G_.select_cols(kept), box})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
