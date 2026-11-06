// polytope_gradients - what the torch tests of the polytope share: inputs and outputs of a result
//
// Syntax:   gradcheck("cart_prod", [](const Inputs &x) { return test::parts(test::halfspaces(x, 0)
//                     .cart_prod(test::halfspaces(x, 2))); }, inputs);
// Sets:     two inputs (A, b) per halfspace polytope, one (V) per vertex polytope; parts are the
//           outputs of a result
// See also: cont_set/zonotope/zonotope_gradients.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/zonotope/zonotope_gradients.h"
#include "cora/cont_set/polytope/polytope.h"
#include "polytope_testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The rows and offsets of a halfspace polytope, or its vertices, as the outputs of a function.
inline Inputs parts(const cora::Polytope &P) {
    if (P.is_h_rep() && !P.is_v_rep()) return {P.A(), P.b()};
    return {P.V()};
}

/// The halfspace polytope made of the two inputs from `first` on.
inline cora::Polytope halfspaces(const Inputs &x, std::size_t first = 0) {
    return {x[first], x[first + 1]};
}

/// A (m, n) with entries away from zero and b (m, 1).
inline std::vector<torch::Tensor> random_halfspaces(int64_t m, int64_t n) {
    const torch::Tensor A = torch::randn({m, n}, torch::kDouble);
    return {A + A.sign() * 0.3, torch::rand({m, 1}, torch::kDouble) + 0.5};
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
