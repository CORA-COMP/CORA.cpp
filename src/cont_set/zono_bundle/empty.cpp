// empty - the empty zonotope bundle of n dimensions, as CORA's zonoBundle.empty
//
// CORA's is one zonotope whose center has no column, which a zonotope of CORA.cpp cannot be.
//
// Syntax:   ZonoBundle bundle = ZonoBundle::empty(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  bundle - the two zonotopes {0} and {e_1}, which have no point in common
// See also: ZonoBundle::ZonoBundle, ConZonotope::empty
//
// CORA: contSet/@zonoBundle/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ZonoBundle ZonoBundle::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "empty: the dimension must be at least 1 (two disjoint points need a coordinate), "
               "got " +
               std::to_string(n);
    });
    std::vector<double> axis(std::size_t(n), 0.0);
    axis[0] = 1.0;
    const Tensor no_generators = Tensor::zeros({n, 0});
    return ZonoBundle({Zonotope(Tensor::zeros({n, 1}), no_generators),
                       Zonotope(Tensor::from_data(axis, {n, 1}), no_generators)});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
