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

#include "contSet/zonotope/zonotope.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce(int order) const {
    if (order < 1) throw std::invalid_argument("cora: the zonotope order must be at least 1");
    const std::vector<int64_t> shape = G.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("cora: reduce takes a single zonotope, not a batch");
    const int64_t n = shape[0], m = shape[1];
    if (m <= order * n) return *this;

    // Which generators stay is decided on the metric; the sets are then built from tensors.
    const Tensor absT = G.abs().transpose();
    const std::vector<double> metric = (absT.sumLast() - absT.maxLast()).data();
    std::vector<int64_t> byMetric(m);
    std::iota(byMetric.begin(), byMetric.end(), 0);
    std::stable_sort(byMetric.begin(), byMetric.end(),
                     [&](int64_t a, int64_t b) { return metric[a] > metric[b]; });

    // The kept generators come first, then one box generator per dimension.
    const int64_t keep = order * n - n;
    const std::vector<int64_t> kept(byMetric.begin(), byMetric.begin() + keep);
    const std::vector<int64_t> rest(byMetric.begin() + keep, byMetric.end());
    const Tensor box = G.selectCols(rest).abs().sumLast().diag();
    return {c, keep == 0 ? box : Tensor::catLast({G.selectCols(kept), box})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
