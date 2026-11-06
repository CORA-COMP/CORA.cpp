// empty - the empty constrained zonotope of n dimensions, as CORA's conZonotope.empty
//
// CORA's empty set has a center without columns, which and_ and cartProd_ turn into a nonempty
// set (docs/cora-differences.md)
//
// Syntax:   ConZonotope cz = ConZonotope::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  cz - the origin without generators and the constraint 0 = 1, which no factor satisfies
// See also: ConZonotope::ConZonotope, EmptySet::empty
//
// CORA: contSet/@conZonotope/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ConZonotope ConZonotope::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return {Tensor::zeros({n, 1}), Tensor::zeros({n, 0}), Tensor::ones({1, 0}),
            Tensor::ones({1, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
