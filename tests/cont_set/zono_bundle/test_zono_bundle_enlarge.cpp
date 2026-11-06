// test_zono_bundle_enlarge - ZonoBundle::enlarge: every member scaled about its own center
//
// CORA: contSet/@zonoBundle/enlarge.m
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

/// A number scales the generators, a column scales the rows; the centers stay.
void scaling(const std::string &b) {
    const Zonotope Z1(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})),
        Z2(column({0.0, 1.0}), Tensor({{1.0}, {2.0}}));
    const ZonoBundle bundle({Z1, Z2});
    const ZonoBundle twice = Set(bundle)->enlarge(2.0).as<ZonoBundle>();
    check(close(twice.zonotopes()[0].generators(), {2.0, 0.0, 0.0, 2.0}, 0) &&
              close(twice.zonotopes()[1].center(), Z2.center(), 0),
          b + ": a number scales every generator");
    const ZonoBundle per_row = Set(bundle)->enlarge(column({1.0, 3.0})).as<ZonoBundle>();
    check(close(per_row.zonotopes()[1].generators(), {1.0, 6.0}, 0), b + ": a column scales rows");
    check(close(per_row.zonotopes()[0].center(), Z1.center(), 0), b + ": the center stays");
    check(Set(bundle)->enlarge(1.0).as<ZonoBundle>().parallel_sets() == 2, b + ": the factor 1");
}

/// Each member is scaled about its own center: member by member it is the zonotope's enlarge.
void members_about_their_centers(const std::string &b) {
    Rng rng(29);
    const ZonoBundle bundle = test::random_zono_bundle(rng, 3, 3, 4);
    const Tensor factor = column({0.5, 2.0, 3.0});
    const ZonoBundle R = Set(bundle)->enlarge(factor).as<ZonoBundle>();
    for (std::size_t i = 0; i < 3; ++i)
        check(close(R.zonotopes()[i].generators(),
                    bundle.zonotopes()[i].enlarge(factor).as<Zonotope>().generators(), 0),
              b + ": member " + std::to_string(i));
}

/// A factor of the wrong size is refused; the same call with the checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({Z, Z});
    check(test::raises([&] { Set(bundle)->enlarge(column({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          "enlarge: the factor has the wrong size");
    ChecksScope off(false);
    check(Set(bundle)->enlarge(column({1.0, 2.0})).as<ZonoBundle>().parallel_sets() == 2,
          "enlarge: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        scaling(b);
        members_about_their_centers(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle enlarge");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
