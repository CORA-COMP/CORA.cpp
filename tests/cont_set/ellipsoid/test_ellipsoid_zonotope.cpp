// test_ellipsoid_zonotope - Ellipsoid zonotope: the parallelotope around or inside an ellipsoid
//
// CORA: contSet/@ellipsoid/zonotope.m
// PROPERTY: contSet/@ellipsoid/zonotope.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

#include <cmath>
#include <limits>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the symmetric root as generators, scaled for the inner one; a point; degenerate.
void zonotope_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Zonotope Z = E.zonotope();
    // layout: #163 step 2
    check(close(cora::detail::cora_center(Z), {1, 2}, 0) &&
              close(cora::detail::cora_generators(Z), {2, 0, 0, 1}, 1e-13),
          b + ": sqrt(Q)");
    // layout: #163 step 2
    check(close(cora::detail::cora_generators(E.zonotope("inner:box")),
                {2 / std::sqrt(2.0), 0, 0, 1 / std::sqrt(2.0)}, 1e-13),
          b + ": inner");
    const Ellipsoid R(Tensor({{2.0, 1.0}, {1.0, 2.0}}));
    // layout: #163 step 2
    const Tensor G = cora::detail::cora_generators(R.zonotope());
    check(close(G.matmul(G.matrix_transpose()), R.Q(), 1e-12) &&
              close(G, G.matrix_transpose(), 1e-12),
          b + ": G is the symmetric root");
    const Ellipsoid flat(Tensor({{1.0, 1.0}, {1.0, 1.0}}), column({0.0, 0.0}), 1e-6);
    // layout: #163 step 2
    check(cora::detail::cora_generators(flat.zonotope()).shape() == Shape({2, 1}),
          b + ": rank 1 gives one generator");
    // layout: #163 step 2
    check(cora::detail::cora_generators(Ellipsoid(Tensor::zeros({2, 2})).zonotope()).shape() ==
              Shape({2, 0}),
          b + ": a point has no generator");
}

/// Stated property: the outer zonotope holds the boundary points, the inner one lies in the
/// ellipsoid.
void zonotope_property(const std::string &b) {
    Rng rng(113);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3);
        check(E.zonotope().contains(test::ellipsoid_points(E, rng, 12, true), "exact", 1e-8),
              b + ": outer holds the points");
        const Zonotope inner = E.zonotope("inner:box");
        check(test::all_inside(E, inner.rand_point(30, rng, "extreme"), 1e-8),
              b + ": inner corners are inside");
    }
}

/// The norm modes, the empty ellipsoid, a bad mode and a batch are refused; checks off works.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor::eye(2));
    check(test::raises([&] { E.zonotope("outer:norm"); }, Error::not_supported),
          "zonotope: norm mode");
    check(test::raises([&] { E.zonotope("outer:cube"); }, Error::wrong_value),
          "zonotope: unknown mode");
    check(test::raises([&] { E.zonotope("outer:box", 0); }, Error::wrong_value),
          "zonotope: generators");
    check(test::raises([] { Ellipsoid::empty(2).zonotope(); }, Error::empty_set),
          "zonotope: empty");
    ChecksScope off(false);
    check(E.zonotope().dim() == 2, "zonotope: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        zonotope_known(b);
        zonotope_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
