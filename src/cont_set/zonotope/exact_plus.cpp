// exact_plus - the Minkowski sum with the coefficients of the first generators shared
//
// The other generators of both follow; the points c + G b + c2 + G2 b with the same b on those r
// generators.
//
// Syntax:   Zres = Z.exact_plus(Z2);   Zres = Z.exact_plus(Z2, k);
// Inputs:   Z2 - zonotope of the same dimension
//           k - at most this many generator pairs are added (default: all that both have)
// Outputs:  Zres - a subset of the plus: centers add, r = min(m, m2, k) generators pairwise
// See also: plus, enclose
//
// CORA: contSet/@zonotope/exactPlus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::exact_plus(const Zonotope &Z2, std::optional<int64_t> k) const {
    detail::Entry entry;
    CORA_CHECK(Z2.dim() == dim() && elem_rank_ == 1 && Z2.elem_rank() == 1,
               Error::dimension_mismatch, [&] {
                   return "exact_plus: the zonotopes differ in dimension, " +
                          std::to_string(dim()) + " and " + std::to_string(Z2.dim());
               });
    CORA_CHECK(!k || *k >= 0, Error::wrong_value,
               [&] { return "exact_plus: k must be nonnegative, got " + std::to_string(*k); });
    const Tensor &G2 = Z2.generators();
    const int64_t m = G_.size(-2), m2 = G2.size(-2);
    const int64_t shared = std::min({m, m2, k.value_or(std::min(m, m2))});
    const Tensor paired = G_.slice(-2, 0, shared) + G2.slice(-2, 0, shared);
    return {c_ + Z2.center(),
            Tensor::concat({paired, G_.slice(-2, shared, m), G2.slice(-2, shared, m2)}, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
