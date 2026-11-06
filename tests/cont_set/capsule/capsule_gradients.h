// capsule_gradients - what the torch tests of the capsule share
//
// Syntax:   gradcheck("lift", [](const Inputs &x) { return test::parts(set(x).lift(...)); },
//                     test::random_capsule_parts(3));
// Sets:     three inputs (c, g, r) per capsule; parts are the three outputs of a result
// See also: cont_set/zonotope/zonotope_gradients.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/zonotope/zonotope_gradients.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The center, generator and radius of a capsule, as the outputs of a function.
inline Inputs parts(const cora::Capsule &C) { return {C.c(), C.generator(), C.r()}; }

/// The capsule made of the three inputs from `first` on.
inline cora::Capsule capsule(const Inputs &x, std::size_t first = 0) {
    return {x[first], x[first + 1], x[first + 2]};
}

/// c (n, 1) and g (n, 1) away from zero, and a radius in [0.3, 1.3].
inline std::vector<torch::Tensor> random_capsule_parts(int64_t n) {
    std::vector<torch::Tensor> out = random_zonotope(n, 1);
    out[1] = out[1].reshape({n, 1});
    out.push_back(0.3 + torch::rand({1, 1}, torch::kDouble));
    return out;
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
