// reduce - fewer generators for a zonotope, as CORA's zonotope.reduce with method "girard"
//
// The longest generators stay; the others are replaced by the box that encloses them. The result
// contains the zonotope and has at most order * n generators (n the dimension).
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

    // Which generators stay is decided on their lengths; the sets are then built from tensors.
    const std::vector<double> len = G.mul(G).transpose().sumLast().data();  // squared lengths
    std::vector<int64_t> byLength(m);
    std::iota(byLength.begin(), byLength.end(), 0);
    std::stable_sort(byLength.begin(), byLength.end(),
                     [&](int64_t a, int64_t b) { return len[a] > len[b]; });

    // The kept generators come first, then one box generator per dimension.
    const int64_t keep = order * n - n;
    const std::vector<int64_t> kept(byLength.begin(), byLength.begin() + keep);
    const std::vector<int64_t> rest(byLength.begin() + keep, byLength.end());
    const Tensor box = G.selectCols(rest).abs().sumLast().diag();
    return {c, keep == 0 ? box : Tensor::catLast({G.selectCols(kept), box})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
