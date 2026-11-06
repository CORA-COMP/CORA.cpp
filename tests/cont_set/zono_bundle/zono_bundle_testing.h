// zono_bundle_testing - what the tests of the zonotope bundle share: random bundles and points
//
// Syntax:   ZonoBundle bundle = test::random_zono_bundle(rng, n, k, m);   Tensor p =
// test::bundle_points(bundle, rng, N); Sets:     the members share one point (given back on
// request), so the intersection is not empty See also: testing.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "testing.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// A bundle of k zonotopes of n dimensions and m generators, all containing `common` (n, 1).
inline cora::ZonoBundle random_zono_bundle(cora::Rng &rng, int n, int k, int m,
                                           cora::Tensor *common = nullptr) {
    std::vector<double> x0(n);
    rng.normal(x0.data(), x0.size(), 1.0);
    const cora::Tensor point = column(x0);
    std::vector<cora::Zonotope> members;
    for (int i = 0; i < k; ++i) {
        std::vector<double> G(std::size_t(n) * m), b(m);
        rng.normal(G.data(), G.size(), 1.0);
        rng.uniform(b.data(), b.size(), -1.0, 1.0);
        const cora::Tensor Gt = cora::Tensor::from_data(G, {n, m});
        // the center is chosen so that x0 = c + G b with |b| <= 1
        members.emplace_back(point - Gt.matmul(column(b)), Gt);
    }
    if (common) *common = point;
    return cora::ZonoBundle(std::move(members));
}

/// The zonotopes [-2, 2] x [-1, 1] and c = (1, 0), G = [1 1; 0 1] (a hexagon in [-1, 3] x [-1, 1]):
/// both contain the origin, and their boxes meet in [-1, 2] x [-1, 1].
inline cora::ZonoBundle two_zonotopes() {
    return cora::ZonoBundle(
        {cora::Zonotope(column({0.0, 0.0}), cora::Tensor({{2.0, 0.0}, {0.0, 1.0}})),
         cora::Zonotope(column({1.0, 0.0}), cora::Tensor({{1.0, 1.0}, {0.0, 1.0}}))});
}

/// Whether the point p (n, 1) lies in every member of the bundle.
inline bool in_every_member(const cora::ZonoBundle &bundle, const cora::Tensor &p) {
    for (const cora::Zonotope &Z : bundle.zonotopes())
        if (!Z.contains(p)) return false;
    return true;
}

/// N points (n, N) of the bundle: the common point first, then by rejection from the first member.
inline cora::Tensor bundle_points(const cora::ZonoBundle &bundle, cora::Rng &rng, int N,
                                  const cora::Tensor &common) {
    std::vector<cora::Tensor> kept{common};
    for (int round = 0; round < 200 && int(kept.size()) < N; ++round) {
        const cora::Tensor draw = bundle.zonotopes().front().rand_point(8, rng);
        for (int k = 0; k < 8 && int(kept.size()) < N; ++k) {
            const cora::Tensor p = draw.slice(-1, k, k + 1);
            if (in_every_member(bundle, p)) kept.push_back(p);
        }
    }
    return cora::Tensor::concat(kept, -1);
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
