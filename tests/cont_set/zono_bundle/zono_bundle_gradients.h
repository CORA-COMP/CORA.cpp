// zono_bundle_gradients - what the torch tests of the zonotope bundle share
//
// Syntax:   gradcheck("cart_prod", [](const Inputs &x) { return test::parts(bundle(x, 0, 2).cart_prod(bundle(x, 4, 2))); },
//                     test::random_bundle_parts(2, 2, 3));
// Sets:     two inputs (c, G) per member; parts are the center and generators of every member
// See also: cont_set/zonotope/zonotope_gradients.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/zonotope/zonotope_gradients.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// The center and generators of every member, as the outputs of a function.
inline Inputs parts(const cora::ZonoBundle &zB) {
    Inputs out;
    for (const cora::Zonotope &Z : zB.zonotopes()) {
        out.push_back(Z.center());
        out.push_back(Z.generators());
    }
    return out;
}

/// The bundle of the k members made of the inputs from `first` on.
inline cora::ZonoBundle bundle(const Inputs &x, std::size_t first, std::size_t k) {
    std::vector<cora::Zonotope> members;
    for (std::size_t i = 0; i < k; ++i) members.emplace_back(x[first + 2 * i], x[first + 2 * i + 1]);
    return cora::ZonoBundle(std::move(members));
}

/// k members (c, G) with centers in [-0.5, 0.5] and generator entries of magnitude in [0.5, 1.5], so
/// the members overlap and no entry is near zero.
inline std::vector<torch::Tensor> random_bundle_parts(int64_t k, int64_t n, int64_t m) {
    std::vector<torch::Tensor> out;
    for (int64_t i = 0; i < k; ++i) {
        const torch::Tensor sign = torch::randint(0, 2, {n, m}, torch::kDouble) * 2 - 1;
        out.push_back(torch::rand({n, 1}, torch::kDouble) - 0.5);
        out.push_back(sign * (0.5 + torch::rand({n, m}, torch::kDouble)));
    }
    return out;
}

/// The inputs of two lists joined.
inline std::vector<torch::Tensor> joined(std::vector<torch::Tensor> a, const std::vector<torch::Tensor> &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
