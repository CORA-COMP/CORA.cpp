// test_con_zonotope_minkowski_difference - ConZonotope::minkowski_difference: {x | x + S in cz}
//
// CORA: contSet/@conZonotope/minkDiff.m
// PROPERTY: contSet/@conZonotope/minkDiff.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: [-2, 2]^2 minus [-1, 1]^2 is [-1, 1]^2 by every type; a point translates.
void minkowski_difference_known(const std::string &b) {
    const ConZonotope big(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 2.0}}));
    const Interval small(column({-1.0, -1.0}), column({1.0, 1.0}));
    const ConZonotope expected(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    for (const char *type : {"exact", "inner:vertices", "inner:Vinod"}) {
        const ConZonotope r = big.minkowski_difference(Set(small), type).as<ConZonotope>();
        check(r.isequal(expected), b + ": the box difference (" + type + ")");
    }
    const Zonotope Z(column({1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(big.minkowski_difference(Set(Z)).as<ConZonotope>().isequal(
              ConZonotope(column({-1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}))),
          b + ": a zonotope with a center");
    check(close(big.minkowski_difference(column({1.0, 1.0})).c(), {-1, -1}, 0), b + ": a point");
    // a subtrahend that does not fit: the exact difference has no point, the Vinod scaling is empty
    const Interval wide(column({-3.0, -3.0}), column({3.0, 3.0}));
    check(big.minkowski_difference(Set(wide), "inner:Vinod")
              .as<ConZonotope>()
              .representsa("emptySet"),
          b + ": too big for the Vinod scaling");
    check(big.minkowski_difference(Set(wide)).as<ConZonotope>().representsa("emptySet"),
          b + ": too big for the exact difference");
}

/// Stated property: the exact difference plus S is inside the set, for sampled points of S.
void minkowski_difference_property(const std::string &b) {
    cora::Rng rng(49);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz =
            test::random_con_zonotope(rng, 2, 4, 1)
                .plus(ConZonotope(column({0.0, 0.0}), Tensor({{4.0, 0.0}, {0.0, 4.0}})));
        const Zonotope S(column({0.0, 0.0}), Tensor({{0.2, 0.1}, {0.0, 0.2}}));
        const ConZonotope r = cz.minkowski_difference(Set(S)).as<ConZonotope>();
        if (r.representsa("emptySet")) continue;
        const Interval sum = r.plus(S).interval(), box = cz.interval();
        check(test::inside_box(sum.inf(), box) && test::inside_box(sum.sup(), box),
              b + ": the difference plus S is inside");
        const Interval inner =
            cz.minkowski_difference(Set(S), "inner:vertices").as<ConZonotope>().interval();
        const Interval hull = r.interval();
        check(test::inside_box(inner.inf(), hull) && test::inside_box(inner.sup(), hull),
              b + ": the inner difference is inside the exact one");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        minkowski_difference_known(b);
        minkowski_difference_property(b);
    });
    const ConZonotope a(column({0.0}), Tensor({{1.0}}));
    const Interval I(column({0.0}), column({1.0}));
    check(test::raises([&] { a.minkowski_difference(Set(I), "other"); }, Error::wrong_value),
          "minkowski_difference: type");
    check(test::raises([&] { a.minkowski_difference(Set(ConZonotope::origin(2))); },
                       Error::dimension_mismatch),
          "minkowski_difference: dimension");
    ChecksScope off(false);
    check(a.minkowski_difference(Set(I)).as<ConZonotope>().dim() == 1,
          "minkowski_difference: checks off");
    return test::finish("con_zonotope minkowski_difference");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
