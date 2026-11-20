// test_zono_bundle_plus - ZonoBundle::plus: the sum of a bundle with a set or a point
//
// CORA: contSet/@zonoBundle/plus.m
// PROPERTY: contSet/@zonoBundle/plus.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a point moves the centers, a zonotope or a box joins its generators to every
/// member, a bundle gives the sum of every pair (the first bundle's index slowest).
void plus_known(const std::string &b) {
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const ZonoBundle moved = zono_bundle.plus(vec({1.0, 2.0}));
    check(close((moved.zonotopes()[0]).center().expand_dims(-1), {1, 2}, 0) &&
              close((moved.zonotopes()[1]).center().expand_dims(-1), {2, 2}, 0),
          b + ": a point moves the centers");
    const Zonotope W(vec({0.0, 0.0}), Tensor({{1.0, 0.0}}));
    check(close((zono_bundle.plus(W).zonotopes()[0]).generators().matrix_transpose(),
                {2, 0, 1, 0, 1, 0}, 0),
          b + ": a zonotope joins its generators");
    const ZonoBundle boxed = zono_bundle.plus(Interval(vec({0.0, 0.0}), vec({2.0, 4.0})));
    check(close((boxed.zonotopes()[0]).center().expand_dims(-1), {1, 2}, 0) &&
              close((boxed.zonotopes()[0]).generators().matrix_transpose(),
                    {2, 0, 1, 0, 0, 1, 0, 2}, 0),
          b + ": a box is added as its zonotope");
    // a bundle is added pair by pair, the first one varying slowest
    const Zonotope W2(vec({5.0, 5.0}), Tensor({{1.0, 1.0}}));
    const ZonoBundle pairs = zono_bundle.plus(ZonoBundle({W, W2}));
    check(pairs.parallel_sets() == 4 &&
              close((pairs.zonotopes()[1]).center().expand_dims(-1), {5, 5}, 0) &&
              close((pairs.zonotopes()[2]).center().expand_dims(-1), {1, 0}, 0) &&
              close((pairs.zonotopes()[3]).center().expand_dims(-1), {6, 5}, 0),
          b + ": a bundle gives the pairs");
    check(zono_bundle.plus(Interval::empty(2)).representsa("emptySet"),
          b + ": the empty box gives the empty set");
    check(Set(zono_bundle)->minus(vec({-1.0, -2.0})).as<ZonoBundle>().parallel_sets() == 2,
          b + ": through the interface");
}

/// Stated property: the sum of a point of the bundle and a point of the box is in the sum.
void plus_property(const std::string &b) {
    Rng rng(62);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 3, &common);
    const Zonotope W = Zonotope::generate_random(rng, 3);
    const Tensor x = test::bundle_points(zono_bundle, rng, 10, common), w = W.rand_point(10, rng);
    check(zono_bundle.plus(W).contains(x + w), b + ": the sums are inside");
    const ZonoBundle other = test::random_zono_bundle(rng, 3, 2, 3, &common);
    const Tensor y = test::bundle_points(other, rng, 10, common);
    check(zono_bundle.plus(other).contains(x + y),
          b + ": the sums of two bundles' points are inside");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        plus_known(b);
        plus_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.plus(vec({1.0})); }, Error::dimension_mismatch),
          "plus: a point of another dimension");
    check(test::raises(
              [&] { zono_bundle.plus(Zonotope(vec({0.0, 0.0, 0.0}), Tensor::zeros({0, 3}))); },
              Error::dimension_mismatch),
          "plus: a zonotope of another dimension");
    ChecksScope off(false);
    check(zono_bundle.plus(vec({1.0, 2.0})).parallel_sets() == 2,
          "plus: the same with the checks off");
    return test::finish("zono_bundle plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
