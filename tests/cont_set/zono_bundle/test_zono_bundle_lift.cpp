// test_zono_bundle_lift - ZonoBundle::lift: every member with its states at given positions
//
// CORA: contSet/@zonoBundle/lift_.m
// PROPERTY: contSet/@zonoBundle/lift_.m
//
// CORA authors: Tobias Ladner
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

/// State i goes to position dims[i]: member by member, as Zonotope::lift.
void permutation(const std::string &b) {
    const Zonotope Z1(column({1.0, 2.0, 3.0}), Tensor({{1.0}, {2.0}, {3.0}})),
        Z2(column({4.0, 5.0, 6.0}), Tensor({{0.0, 1.0}, {1.0, 0.0}, {2.0, 2.0}}));
    const ZonoBundle bundle({Z1, Z2});
    const std::vector<int64_t> dims{2, 0, 1};
    const ZonoBundle R = bundle.lift(3, dims);
    check(close(R.zonotopes()[0].center(), {2.0, 3.0, 1.0}, 0), b + ": state 0 goes to position 2");
    check(close(R.zonotopes()[1].generators(), Z2.lift(3, dims).generators(), 0), b + ": member 2");
    const Set S = Set(bundle)->lift(3, dims);
    check(S.is<ZonoBundle>() &&
              close(S.as<ZonoBundle>().zonotopes()[0].center(), {2.0, 3.0, 1.0}, 0),
          b + ": through the set interface");
    check(close(Set(bundle)->lift(3).as<ZonoBundle>().zonotopes()[0].center(), Z1.center(), 0),
          b + ": the identity by default");
}

/// Stated property: the permuted point of a point of the bundle is a point of the lifted bundle.
void permuted_points(const std::string &b) {
    Rng rng(30);
    Tensor x0;
    const ZonoBundle bundle = test::random_zono_bundle(rng, 3, 2, 3, &x0);
    const ZonoBundle R = bundle.lift(3, {1, 2, 0});
    const Tensor points = test::bundle_points(bundle, rng, 6, x0);
    for (int k = 0; k < 6; ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        // state i at position dims[i]: the new coordinates are (p1', p2', p0')... in position order
        const Tensor q =
            Tensor::concat({p.slice(-2, 2, 3), p.slice(-2, 0, 1), p.slice(-2, 1, 2)}, -2);
        check(test::in_every_member(R, q), b + ": a permuted point");
    }
}

/// New dimensions are refused (they would be unbounded), so are bad positions; the same call with
/// the checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({Z, Z});
    check(test::raises([&] { bundle.lift(3, {0, 1}); }, Error::not_defined), "lift: N > dimension");
    check(test::raises([&] { bundle.lift(2, {0, 2}); }, Error::wrong_value),
          "lift: position out of range");
    check(test::raises([&] { bundle.lift(2, {1, 1}); }, Error::wrong_value),
          "lift: a position twice");
    ChecksScope off(false);
    check(bundle.lift(2, {1, 0}).parallel_sets() == 2, "lift: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        permutation(b);
        permuted_points(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
