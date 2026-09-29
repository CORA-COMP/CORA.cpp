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
#include <cmath>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The Euclidean length of every column of the row-major matrix G (n, m).
std::vector<double> aux_lengths(const std::vector<double> &G, int64_t n, int64_t m) {
    std::vector<double> len(m, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) len[j] += G[i * m + j] * G[i * m + j];
    for (double &l : len) l = std::sqrt(l);
    return len;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce(int order) const {
    if (order < 1) throw std::invalid_argument("cora: the zonotope order must be at least 1");
    const std::vector<int64_t> shape = G.shape();
    if (shape.size() != 2)
        throw std::invalid_argument("cora: reduce takes a single zonotope, not a batch");
    const int64_t n = shape[0], m = shape[1];
    if (m <= order * n) return *this;

    const std::vector<double> data = G.data();
    const std::vector<double> len = aux_lengths(data, n, m);
    std::vector<int64_t> byLength(m);
    std::iota(byLength.begin(), byLength.end(), 0);
    std::sort(byLength.begin(), byLength.end(),
              [&](int64_t a, int64_t b) { return len[a] > len[b]; });

    // The kept generators come first, then one box generator per dimension.
    const int64_t keep = order * n - n;
    std::vector<double> reduced(n * (keep + n), 0.0);
    for (int64_t j = 0; j < m; ++j) {
        const int64_t rank = std::find(byLength.begin(), byLength.end(), j) - byLength.begin();
        for (int64_t i = 0; i < n; ++i) {
            if (rank < keep) reduced[i * (keep + n) + rank] = data[i * m + j];
            else reduced[i * (keep + n) + keep + i] += std::abs(data[i * m + j]);
        }
    }
    return {c, Tensor::like(G, reduced, {n, keep + n})};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
