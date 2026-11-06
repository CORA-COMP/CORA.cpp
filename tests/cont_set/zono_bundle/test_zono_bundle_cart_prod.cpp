// test_zono_bundle_cart_prod - ZonoBundle::cart_prod: the product with a bundle, a set or a point
//
// CORA: contSet/@zonoBundle/cartProd_.m
// PROPERTY: contSet/@zonoBundle/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two bundles give every pair of members, the first bundle's index outer.
void pairs_of_members(const std::string &b) {
    const Zonotope A1(column({1.0, 2.0}), Tensor({{1.0}, {3.0}})),
        A2(column({0.0, 1.0}), Tensor({{2.0}, {1.0}})), B1(column({5.0}), Tensor({{6.0, 7.0}})),
        B2(column({9.0}), Tensor({{1.0, 1.0}}));
    const ZonoBundle P = ZonoBundle({A1, A2}).cart_prod(ZonoBundle({B1, B2}));
    check(P.parallel_sets() == 4 && P.dim() == 3, b + ": four members of dimension 3");
    const std::vector<Zonotope> expected{A1.cart_prod(B1), A1.cart_prod(B2), A2.cart_prod(B1),
                                         A2.cart_prod(B2)};
    for (std::size_t i = 0; i < 4; ++i)
        check(close(P.zonotopes()[i].center(), expected[i].center(), 0) &&
                  close(P.zonotopes()[i].generators(), expected[i].generators(), 0),
              b + ": pair " + std::to_string(i));
}

/// A zonotope, a box and a point multiply every member; a point may come first.
void one_set_against_every_member(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {3.0}}));
    const ZonoBundle bundle({Z, Z});
    const Interval I(column({0.0}), column({2.0}));
    check(bundle.cart_prod(I).parallel_sets() == 2 && bundle.cart_prod(I).dim() == 3,
          b + ": a box");
    check(close(bundle.cart_prod(column({7.0})).zonotopes()[1].center(), {1.0, 2.0, 7.0}, 0),
          b + ": a point behind");
    check(close(ZonoBundle::cart_prod(column({7.0}), bundle).zonotopes()[0].center(),
                {7.0, 1.0, 2.0}, 0),
          b + ": a point in front");
    check(bundle.cart_prod(Z).dim() == 4, b + ": a zonotope");
    const Set S = cart_prod(Set(bundle), Set(Z));
    check(S.is<ZonoBundle>() && S.as<ZonoBundle>().dim() == 4, b + ": through the registry");
}

/// Stated property: the pair (x, y) of a point of each factor is a point of the product.
void points_of_the_product(const std::string &b) {
    Rng rng(27);
    Tensor x0, y0;
    const ZonoBundle A = test::random_zono_bundle(rng, 2, 2, 3, &x0),
                     B = test::random_zono_bundle(rng, 3, 2, 3, &y0);
    const ZonoBundle P = A.cart_prod(B);
    const Tensor xs = test::bundle_points(A, rng, 6, x0), ys = test::bundle_points(B, rng, 6, y0);
    for (int k = 0; k < 6; ++k)
        check(test::in_every_member(
                  P, Tensor::concat({xs.slice(-1, k, k + 1), ys.slice(-1, k, k + 1)}, -2)),
              b + ": a pair of points");
}

/// A point that is no column, another batch shape and an empty box are refused; the same call with
/// the checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({Z});
    check(test::raises([&] { bundle.cart_prod(Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
          "cart_prod: a point must be a column");
    check(test::raises([&] { ZonoBundle::cart_prod(Tensor({{1.0, 2.0}}), bundle); },
                       Error::dimension_mismatch),
          "cart_prod: a point in front must be a column");
    check(
        test::raises([&] { bundle.cart_prod(Zonotope::stack({Z, Z})); }, Error::dimension_mismatch),
        "cart_prod: another batch shape");
    check(test::raises([&] { bundle.cart_prod(Interval::empty(1)); }, Error::not_supported),
          "cart_prod: an empty box");
    ChecksScope off(false);
    check(bundle.cart_prod(bundle).dim() == 4, "cart_prod: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        pairs_of_members(b);
        one_set_against_every_member(b);
        points_of_the_product(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
