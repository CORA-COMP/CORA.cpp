// con_zonotope_gradients - what the torch tests of the constrained zonotope share
//
// Syntax:   gradcheck("cart_prod", [](const Inputs &x) { return test::parts(set(x,
// 0).cart_prod(set(x, 4))); },
//                     test::random_con_zonotope_parts(2, 3, 1));
// Sets:     four inputs (c, G, A, b) per set; parts are the four outputs of a result
// See also: cont_set/zonotope/zonotope_gradients.h

#pragma once

#include "cont_set/zonotope/zonotope_gradients.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"

namespace test {

/// The center, generators, constraint matrix and offset of a set, as the outputs of a function.
inline Inputs parts(const cora::ConZonotope &cz) {
    return {cz.c(), cz.generators(), cz.A(), cz.b()};
}

/// The set made of the four inputs from `first` on.
inline cora::ConZonotope set(const Inputs &x, std::size_t first = 0) {
    return {x[first], x[first + 1], x[first + 2], x[first + 3]};
}

/// c (n, 1), G (n, m), A (p, m), b (p, 1) away from zero.
inline std::vector<torch::Tensor> random_con_zonotope_parts(int64_t n, int64_t m, int64_t p) {
    std::vector<torch::Tensor> out = random_zonotope(n, m);
    const std::vector<torch::Tensor> rest = random_zonotope(p, m);
    out.push_back(rest[1]);
    out.push_back(rest[0]);
    return out;
}

} // namespace test
