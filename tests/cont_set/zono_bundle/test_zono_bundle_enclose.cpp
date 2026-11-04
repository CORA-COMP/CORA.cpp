// test_zono_bundle_enclose - ZonoBundle::enclose: the members enclosed one by one
//
// CORA: contSet/@zonoBundle/enclose.m
// PROPERTY: contSet/@zonoBundle/enclose.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Member i of the result is the enclosure of member i of both bundles; a zonotope meets every one.
void member_by_member(const std::string &b) {
    const Zonotope A1(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})),
        A2(column({1.0, 0.0}), Tensor({{1.0}, {1.0}})),
        B1(column({4.0, 0.0}), Tensor({{1.0}, {0.5}})),
        B2(column({0.0, 3.0}), Tensor({{2.0, 0.0}, {0.0, 1.0}}));
    const ZonoBundle R = ZonoBundle({A1, A2}).enclose(ZonoBundle({B1, B2}));
    check(R.parallel_sets() == 2, b + ": two members");
    check(close(R.zonotopes()[0].generators(), A1.enclose(B1).generators(), 0) &&
              close(R.zonotopes()[1].center(), A2.enclose(B2).center(), 0),
          b + ": the members enclosed pairwise");
    const ZonoBundle with_zonotope = ZonoBundle({A1, A2}).enclose(B1);
    check(close(with_zonotope.zonotopes()[1].generators(), A2.enclose(B1).generators(), 0),
          b + ": a zonotope against every member");
    const Set S = Set(ZonoBundle({A1, A2})).get().enclose(Set(ZonoBundle({B1, B2})));
    check(S.is<ZonoBundle>() && S.as<ZonoBundle>().parallel_sets() == 2,
          b + ": through the set interface");
}

/// The form with a map and a shift takes M Z_i + Zplus; a bundle as shift is replaced by the
/// zonotope of its box (any number of members).
void with_a_map(const std::string &b) {
    const Zonotope A1(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})),
        A2(column({1.0, 0.0}), Tensor({{1.0}, {1.0}})),
        Zp(column({0.5, 0.5}), Tensor({{0.1}, {0.0}}));
    const Tensor M({{0.0, -1.0}, {1.0, 0.0}});
    const ZonoBundle bundle({A1, A2});
    const ZonoBundle R = bundle.enclose(M, Zp);
    check(close(R.zonotopes()[1].generators(), A2.enclose(M, Zp).generators(), 0),
          b + ": M Z + Zplus");
    const ZonoBundle shift({Zp, Zp, Zp});
    const ZonoBundle R2 = bundle.enclose(M, shift);
    check(R2.parallel_sets() == 2 && close(R2.zonotopes()[0].center(),
                                           A1.enclose(M, shift.interval().zonotope()).center(), 0),
          b + ": a bundle shifts as its box");
}

/// Stated property: a convex combination of a point of the bundle and a point of its image is in
/// the enclosure.
void convex_combinations(const std::string &b) {
    Rng rng(28);
    Tensor x0, y0;
    const ZonoBundle A = test::random_zono_bundle(rng, 3, 2, 3, &x0),
                     B = test::random_zono_bundle(rng, 3, 2, 3, &y0);
    const ZonoBundle R = A.enclose(B);
    const Tensor xs = test::bundle_points(A, rng, 5, x0), ys = test::bundle_points(B, rng, 5, y0);
    for (int k = 0; k < 5; ++k)
        for (const double lambda : {0.0, 0.3, 0.5, 1.0}) {
            const Tensor p =
                xs.slice(-1, k, k + 1) * lambda + ys.slice(-1, k, k + 1) * (1 - lambda);
            check(R.contains(p, "exact:zonotope", 1e-9), b + ": a point of the hull");
        }
}

/// Bundles of other sizes, a map of another size and a box are refused; the same call with the
/// checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle one({Z}), two({Z, Z});
    check(test::raises([&] { one.enclose(two); }, Error::wrong_value), "enclose: member counts");
    check(test::raises([&] { one.enclose(Tensor({{1.0}}), Z); }, Error::dimension_mismatch),
          "enclose: M must be square of the dimension");
    check(test::raises([&] { one.enclose(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "enclose: another dimension");
    check(
        test::raises(
            [&] { Set(one).get().enclose(Set(Interval(column({0.0, 0.0}), column({1.0, 1.0})))); },
            Error::noops),
        "enclose: a box is not a class of CORA's bundle enclose");
    ChecksScope off(false);
    check(one.enclose(Z).parallel_sets() == 1, "enclose: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        member_by_member(b);
        with_a_map(b);
        convex_combinations(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
