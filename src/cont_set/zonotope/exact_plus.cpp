// exact_plus - the Minkowski sum with the coefficients of the first generators shared
//
// Syntax:   Zres = Z.exact_plus(Z2);   Zres = Z.exact_plus(Z2, k);
// Inputs:   Z2 - zonotope of the same dimension;  k - at most this many generator pairs are added
//           (default: all that both have)
// Outputs:  Zres - the centers add, the first r = min(m, m2, k) generators are added pairwise, and
//           the others of both follow; the points c + G b + c2 + G2 b with the same b on those r
//           generators, a subset of the plus
// See also: plus, enclose
//
// CORA: contSet/@zonotope/exactPlus.m

#include "cora/cont_set/zonotope/zonotope.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::exact_plus(const Zonotope &Z2, std::optional<int64_t> k) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "exact_plus: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(Z2.dim());
    });
    CORA_CHECK(!k || *k >= 0, Error::wrong_value,
               [&] { return "exact_plus: k must be nonnegative, got " + std::to_string(*k); });
    const Tensor &G2 = Z2.generators();
    const int64_t m = G_.shape().back(), m2 = G2.shape().back();
    const int64_t shared = std::min({m, m2, k.value_or(std::min(m, m2))});
    const Tensor paired = G_.slice(-1, 0, shared) + G2.slice(-1, 0, shared);
    return {c_ + Z2.center(),
            Tensor::concat({paired, G_.slice(-1, shared, m), G2.slice(-1, shared, m2)}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
