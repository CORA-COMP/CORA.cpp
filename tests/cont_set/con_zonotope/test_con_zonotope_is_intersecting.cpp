// test_con_zonotope_is_intersecting - ConZonotope::is_intersecting: whether two sets meet
//
// CORA: contSet/@conZonotope/isIntersecting_.m

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// Hand-computed: the segment from (0, 1) to (1, 0) meets the box [0.4, 2]^2 and misses [0.6, 2]^2,
/// whose box around it still overlaps ('approx').
void is_intersecting_known(const std::string &b) {
    const ConZonotope seg(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                          column({1.0}));
    const Interval meets(column({0.4, 0.4}), column({2.0, 2.0}));
    const Interval misses(column({0.6, 0.6}), column({2.0, 2.0}));
    check(seg.is_intersecting(meets) && !seg.is_intersecting(misses), b + ": a box");
    check(seg.is_intersecting(misses, "approx"), b + ": the boxes overlap");
    check(!seg.is_intersecting(Interval(column({2.0, 2.0}), column({3.0, 3.0})), "approx"),
          b + ": the boxes are apart");
    const Zonotope Z(column({1.0, 1.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(seg.is_intersecting(Z) && !seg.is_intersecting(Zonotope(column({3.0, 3.0}), Z.generators())),
          b + ": a zonotope");
    check(seg.is_intersecting(seg) && !seg.is_intersecting(seg.plus(column({2.0, 0.0}))),
          b + ": a constrained zonotope");
    check(seg.is_intersecting(column({0.5, 0.5})) && !seg.is_intersecting(column({1.0, 1.0})),
          b + ": a point");
    check(!seg.is_intersecting(ConZonotope::empty(2)), b + ": the empty set");
    check(Set(seg)->dim() == 2 && cora::is_intersecting(seg, meets), b + ": through the registry");
}

} // namespace

int main() {
    test::for_each_backend(is_intersecting_known);
    const ConZonotope a(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.is_intersecting(a, "fast"); }, Error::wrong_value), "is_intersecting: type");
    check(test::raises([&] { a.is_intersecting(a, "exact", -1.0); }, Error::wrong_value),
          "is_intersecting: tol");
    check(test::raises([&] { a.is_intersecting(ConZonotope::origin(2)); }, Error::dimension_mismatch),
          "is_intersecting: dimension");
    ChecksScope off(false);
    check(a.is_intersecting(a), "is_intersecting: the same with the checks off");
    return test::finish("con_zonotope is_intersecting");
}
