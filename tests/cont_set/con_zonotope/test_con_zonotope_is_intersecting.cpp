// test_con_zonotope_is_intersecting - ConZonotope::is_intersecting: whether two sets meet
//
// CORA: contSet/@conZonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::column;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the segment from (0, 1) to (1, 0) meets the box [0.4, 2]^2 and misses [0.6, 2]^2,
/// whose box around it still overlaps ('approx').
void is_intersecting_known(const std::string &b) {
    const ConZonotope seg(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                          Tensor({{1.0, 1.0}}), vec({1.0}));
    const Interval meets(vec({0.4, 0.4}), vec({2.0, 2.0}));
    const Interval misses(vec({0.6, 0.6}), vec({2.0, 2.0}));
    check(seg.is_intersecting(meets) && !seg.is_intersecting(misses), b + ": a box");
    check(seg.is_intersecting(misses, "approx"), b + ": the boxes overlap");
    check(!seg.is_intersecting(Interval(vec({2.0, 2.0}), vec({3.0, 3.0})), "approx"),
          b + ": the boxes are apart");
    // a zonotope and a constrained zonotope, then a point and the empty set
    const Zonotope Z(vec({1.0, 1.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(seg.is_intersecting(Z) &&
              !seg.is_intersecting(Zonotope(vec({3.0, 3.0}), Z.generators())),
          b + ": a zonotope");
    check(seg.is_intersecting(seg) && !seg.is_intersecting(seg.plus(vec({2.0, 0.0}))),
          b + ": a constrained zonotope");
    check(seg.is_intersecting(vec({0.5, 0.5})) && !seg.is_intersecting(vec({1.0, 1.0})),
          b + ": a point");
    check(!seg.is_intersecting(ConZonotope::empty(2)), b + ": the empty set");
    check(Set(seg)->dim() == 2 && cora::is_intersecting(seg, meets), b + ": through the registry");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(is_intersecting_known);
    const ConZonotope a(vec({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.is_intersecting(a, "fast"); }, Error::wrong_value),
          "is_intersecting: type");
    check(test::raises([&] { a.is_intersecting(a, "exact", -1.0); }, Error::wrong_value),
          "is_intersecting: tol");
    check(
        test::raises([&] { a.is_intersecting(ConZonotope::origin(2)); }, Error::dimension_mismatch),
        "is_intersecting: dimension");
    ChecksScope off(false);
    check(a.is_intersecting(a), "is_intersecting: the same with the checks off");
    return test::finish("con_zonotope is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
