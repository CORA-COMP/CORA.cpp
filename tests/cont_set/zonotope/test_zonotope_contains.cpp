// test_zonotope_contains - zonotope contains: a point, a zonotope or a box in a zonotope
//
// CORA: contSet/@zonotope/contains_.m
// PROPERTY: contSet/@zonotope/contains_.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Adrian Kulmburg, Ivan Brkan
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <chrono>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The diamond |x| + |y| <= 2.
Zonotope diamond() { return Zonotope(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}})); }

/// Points: inside, on the boundary, outside, and the tolerance that relaxes the boundary.
void points_in_a_diamond(const std::string &b) {
    const Zonotope Z = diamond();
    check(Z.contains(column({0.0, 0.0})), b + ": the center");
    check(Z.contains(column({1.0, 1.0})), b + ": the boundary");
    check(Z.contains(Tensor({{0.5, 2.0, -1.0}, {0.5, 0.0, 0.5}})), b + ": three points, all in");
    check(!Z.contains(Tensor({{0.5, 1.5}, {0.5, 1.5}})), b + ": one of two points is out");
    check(!Z.contains(column({1.0 + 1e-10, 1.0})), b + ": just outside, default tol");
    check(Z.contains(column({1.0 + 1e-10, 1.0}), "exact", 1e-6), b + ": just outside, tol 1e-6");
    check(Z.contains(column({1.0, 1.0}), "exact:venum") && Z.contains(column({1.0, 1.0}), "approx"),
          b + ": the other point methods");
}

/// The special cases of CORA's contains_: a point, a box, and a flat zonotope.
void special_zonotopes(const std::string &b) {
    const Zonotope point(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}));
    check(point.contains(column({1.0, 2.0})) && !point.contains(column({1.0, 2.5})),
          b + ": a point");
    check(point.contains(Zonotope(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}))),
          b + ": a point holds itself");
    check(!point.contains(diamond()), b + ": a point holds no set");
    const Zonotope box(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(box.contains(column({1.0, -1.0})) && !box.contains(column({1.1, 0.0})), b + ": a box");
    check(box.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.0}))),
          b + ": a box holds a box");
    check(!box.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.5}))),
          b + ": a box, not a larger one");
    const Zonotope segment(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(segment.contains(column({0.5, 0.5})), b + ": a segment holds its points");
    check(!segment.contains(column({0.5, 0.6})), b + ": and no other");
}

/// Sets: a zonotope in a zonotope by both exact methods and by Sadraddini and Tedrake.
void sets_in_the_diamond(const std::string &b) {
    const Zonotope Z = diamond();
    const Zonotope small = 0.5 * Z, large = 1.2 * Z;
    for (const std::string method :
         {"exact", "exact:venum", "exact:polymax", "approx", "approx:st"}) {
        check(Z.contains(small, method), b + ": a half in the diamond, " + method);
        check(!Z.contains(large, method), b + ": 1.2 times is not in, " + method);
    }
    check(Z.contains(Z), b + ": a zonotope holds itself");
    check(Z.contains(Interval(column({-0.5, -0.5}), column({0.5, 0.5}))), b + ": a box inside");
    check(!Z.contains(Interval(column({-1.5, -1.5}), column({1.5, 1.5}))), b + ": a box too large");
    check(Z.contains(small + column({0.5, 0.0})) && !Z.contains(small + column({1.5, 0.0})),
          b + ": shifted");
    check(contains(Z, small) && !contains(Z, large), b + ": through the registry");
    check(contains(Z, Interval(column({-0.5, -0.5}), column({0.5, 0.5}))),
          b + ": a box through the registry");
}

/// Stated property: c + s (Z' - c) is in Z for s < 1 and not for s > 1, for random zonotopes, and
/// every sampled point of Z is in Z.
void scaled_copies(const std::string &b) {
    cora::Rng rng(26);
    for (int trial = 0; trial < 6; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 5);
        const Tensor &c = Z.center();
        const Zonotope inside(c, Z.generators() * 0.9), outside(c, Z.generators() * 1.1);
        for (const std::string method : {"exact", "approx"}) {
            check(Z.contains(inside, method), b + ": 0.9 Z in Z, " + method);
            check(!Z.contains(outside, method), b + ": 1.1 Z not in Z, " + method);
        }
        check(Z.contains(Z.rand_point(20, rng)), b + ": sampled points are in");
        check(Z.contains(Z.rand_point(20, rng, "extreme")), b + ": extreme points are in");
    }
}

/// Polymax ("exact" below dimension 4) agrees with the vertex search on random zonotopes in two
/// and three dimensions, scaled and shifted, and holds a flat zonotope's own segment.
void polymax_agrees_with_venum(const std::string &b) {
    cora::Rng rng(9);
    for (int trial = 0; trial < 8; ++trial) {
        const int64_t n = 2 + trial % 2;
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 4 + trial);
        const Zonotope S = Zonotope::generate_random(rng, n, std::nullopt, 3);
        // S scaled into the radius of Z, then moved, so both answers occur
        const Zonotope T(Z.center() + S.center() * 0.3, S.generators() * (0.15 * (trial + 1)));
        check(Z.contains(T, "exact:polymax") == Z.contains(T, "exact:venum"),
              b + ": polymax and venum agree, trial " + std::to_string(trial));
        check(Z.contains(T) == Z.contains(T, "exact:polymax"), b + ": exact is polymax");
    }
    const Zonotope segment(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const Zonotope half(column({0.0, 0.0}), Tensor({{0.5}, {0.5}}));
    check(segment.contains(half, "exact:polymax"), b + ": a flat zonotope holds half of itself");
    check(!segment.contains(half + column({0.0, 0.1}), "exact:polymax"), b + ": not off its line");
}

/// A zonogon of 4000 generators (the platoon benchmark's projections have thousands): polymax
/// takes one matrix product where the vertex search would not finish.
void large_zonogon(const std::string &b) {
    cora::Rng rng(4);
    const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 4000);
    const Tensor &c = Z.center();
    const auto start = std::chrono::steady_clock::now();
    check(Z.contains(Zonotope(c, Z.generators() * 0.99)), b + ": 0.99 Z in Z, 4000 generators");
    check(!Z.contains(Zonotope(c, Z.generators() * 1.01)), b + ": 1.01 Z not in Z");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    check(seconds < 10.0, b + ": 4000 generators in " + std::to_string(seconds) + " s");
}

/// Wrong operands and methods are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z = diamond();
    check(test::raises([&] { Z.contains(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "contains: a point of dimension 3");
    check(test::raises([&] { Z.contains(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "contains: a zonotope of dimension 1");
    check(test::raises([&] { Z.contains(column({0.0, 0.0}), "exact", -1.0); }, Error::wrong_value),
          "contains: tol < 0");
    check(test::raises([&] { Z.contains(column({0.0, 0.0}), "simplex"); }, Error::no_specific_alg),
          "contains: unknown method");
    check(test::raises([&] { Z.contains(Z, "approx:stDual"); }, Error::not_supported),
          "contains: approx:stDual");
    check(test::raises([&] { Z.contains(Z, "sampling"); }, Error::not_supported),
          "contains: sampling");
    check(test::raises([&] { Zonotope::stack({Z, Z}).contains(Z); }, Error::not_supported),
          "contains: a batch");
    const bool on = Z.contains(0.5 * Z);
    ChecksScope off(false);
    // the same with the checks off
    check(Z.contains(0.5 * Z) == on, "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_in_a_diamond(b);
        special_zonotopes(b);
        sets_in_the_diamond(b);
        scaled_copies(b);
        polymax_agrees_with_venum(b);
        large_zonogon(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
