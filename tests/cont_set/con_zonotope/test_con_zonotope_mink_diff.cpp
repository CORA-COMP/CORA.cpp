// test_con_zonotope_mink_diff - ConZonotope::mink_diff: {x | x + S in cz}
//
// CORA: contSet/@conZonotope/minkDiff.m
// PROPERTY: contSet/@conZonotope/minkDiff.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: [-2, 2]^2 minus [-1, 1]^2 is [-1, 1]^2 by every type; a point translates.
void mink_diff_known(const std::string &b) {
    const ConZonotope big(vec({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 2.0}}));
    const Interval small(vec({-1.0, -1.0}), vec({1.0, 1.0}));
    const ConZonotope expected(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    for (const char *type : {"exact", "inner:vertices", "inner:Vinod"}) {
        const ConZonotope r = big.mink_diff(Set(small), type).as<ConZonotope>();
        check(r.isequal(expected), b + ": the box difference (" + type + ")");
    }
    const Zonotope Z(vec({1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(big.mink_diff(Set(Z)).as<ConZonotope>().isequal(
              ConZonotope(vec({-1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}))),
          b + ": a zonotope with a center");
    check(close(big.mink_diff(vec({1.0, 1.0})).c().expand_dims(-1), {-1, -1}, 0), b + ": a point");
    // a subtrahend that does not fit: the exact difference has no point, the Vinod scaling is empty
    const Interval wide(vec({-3.0, -3.0}), vec({3.0, 3.0}));
    check(big.mink_diff(Set(wide), "inner:Vinod").as<ConZonotope>().representsa("emptySet"),
          b + ": too big for the Vinod scaling");
    check(big.mink_diff(Set(wide)).as<ConZonotope>().representsa("emptySet"),
          b + ": too big for the exact difference");
}

/// Stated property: the exact difference plus S is inside the set, for sampled points of S.
void mink_diff_property(const std::string &b) {
    cora::Rng rng(49);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz =
            test::random_con_zonotope(rng, 2, 4, 1)
                .plus(ConZonotope(vec({0.0, 0.0}), Tensor({{4.0, 0.0}, {0.0, 4.0}})));
        const Zonotope S(vec({0.0, 0.0}), Tensor({{0.2, 0.0}, {0.1, 0.2}}));
        const ConZonotope r = cz.mink_diff(Set(S)).as<ConZonotope>();
        if (r.representsa("emptySet")) continue;
        const Interval sum = r.plus(S).interval(), box = cz.interval();
        check(test::inside_box(sum.inf(), box) && test::inside_box(sum.sup(), box),
              b + ": the difference plus S is inside");
        const Interval inner = cz.mink_diff(Set(S), "inner:vertices").as<ConZonotope>().interval();
        const Interval hull = r.interval();
        check(test::inside_box(inner.inf(), hull) && test::inside_box(inner.sup(), hull),
              b + ": the inner difference is inside the exact one");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        mink_diff_known(b);
        mink_diff_property(b);
    });
    const ConZonotope a(vec({0.0}), Tensor({{1.0}}));
    const Interval I(vec({0.0}), vec({1.0}));
    check(test::raises([&] { a.mink_diff(Set(I), "other"); }, Error::wrong_value),
          "mink_diff: type");
    check(
        test::raises([&] { a.mink_diff(Set(ConZonotope::origin(2))); }, Error::dimension_mismatch),
        "mink_diff: dimension");
    // a batch of constrained zonotopes is refused
    check(test::raises(
              [&] {
                  ConZonotope(Tensor::zeros({2, 1}), Tensor::ones({2, 1, 1}).matrix_transpose())
                      .mink_diff(Set(I));
              },
              Error::not_supported),
          "mink_diff: a batch");
    ChecksScope off(false);
    check(a.mink_diff(Set(I)).as<ConZonotope>().dim() == 1, "mink_diff: checks off");
    return test::finish("con_zonotope mink_diff");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
