// test_zonotope_is_intersecting - zonotope is_intersecting: whether a zonotope meets another set
//
// CORA: contSet/@zonotope/isIntersecting_.m
// PROPERTY: contSet/@zonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: squares apart, touching, within tol; a diamond and a box near its corner, which
/// the boxes around them (type "approx") cannot tell apart.
void intersecting_known(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const auto square_at = [](double x) {
        return Zonotope(column({x, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    };
    check(Z.is_intersecting(square_at(1.5)), b + ": overlapping squares");
    check(Z.is_intersecting(square_at(2.0)), b + ": touching squares");
    check(!Z.is_intersecting(square_at(3.0)), b + ": squares apart");
    check(!Z.is_intersecting(square_at(2.0 + 1e-6)), b + ": apart by 1e-6 at the default tol");
    check(Z.is_intersecting(square_at(2.0 + 1e-6), "exact", 1e-5), b + ": and within tol 1e-5");

    const Zonotope diamond(column({0.0, 0.0}), Tensor({{0.5, 0.5}, {0.5, -0.5}}));
    const Interval corner(column({0.6, 0.6}), column({1.0, 1.0}));
    check(!diamond.is_intersecting(corner), b + ": the diamond |x| + |y| <= 1 misses [0.6, 1]^2");
    check(diamond.is_intersecting(corner, "approx"),
          b + ": their boxes meet, so 'approx' says true");
    check(diamond.is_intersecting(Interval(column({0.2, 0.2}), column({1.0, 1.0}))),
          b + ": a box that cuts the diamond");
    check(!diamond.is_intersecting(Zonotope(column({0.9, 0.9}), Tensor({{0.1}, {0.1}}))),
          b + ": a segment beyond the corner");

    check(diamond.is_intersecting(column({0.5, 0.5})), b + ": a point on the boundary");
    // points and registry
    check(!diamond.is_intersecting(column({0.6, 0.6})), b + ": a point outside");
    check(!diamond.is_intersecting(Tensor({{0.0, 0.6}, {0.0, 0.6}})),
          b + ": points: all must be in");
    check(is_intersecting(Z, square_at(1.5)), b + ": through the registry");
    check(!is_intersecting(Z, Interval(column({3.0, 0.0}), column({4.0, 1.0}))),
          b + ": zonotope and box through the registry");
}

/// Stated property: symmetric; true when a sampled point of one lies in the other; false when a
/// separating direction exists.
void intersecting_property(const std::string &b) {
    cora::Rng rng(26);
    for (int k = 0; k < 30; ++k) {
        const Zonotope Z1 = Zonotope::generate_random(rng, 3, std::nullopt, 4);
        std::vector<double> shift(3);
        rng.uniform(shift.data(), 3, -15.0, 15.0);
        const Zonotope Z2 = Zonotope::generate_random(rng, 3, std::nullopt, 3) + column(shift);
        const bool res = Z1.is_intersecting(Z2);
        check(res == Z2.is_intersecting(Z1), b + ": symmetric");
        if (Z1.contains(Z2.rand_point(5, rng), "exact", 0.0))
            check(res, b + ": a point of Z2 is in Z1");
        for (int j = 0; j < 100; ++j) {
            const std::vector<double> d = test::random_direction(rng, 3);
            std::vector<double> minus(d);
            for (double &x : minus) x = -x;
            if (test::support(Z1, d) + test::support(Z2, minus) < 0) check(!res, b + ": separated");
        }
        // approx never misses an intersection
        if (res)
            check(Z1.is_intersecting(Z2, "approx"), b + ": approx never misses an intersection");
    }
}

/// The refusals, and the call with the checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {0.0}})),
        Z1(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Z.is_intersecting(Z1); }, Error::dimension_mismatch),
          "is_intersecting: dimensions");
    check(test::raises([&] { Z.is_intersecting(column({1.0})); }, Error::dimension_mismatch),
          "is_intersecting: points");
    check(test::raises([&] { Z.is_intersecting(Z, "other"); }, Error::wrong_value),
          "is_intersecting: type");
    check(test::raises([&] { Z.is_intersecting(Z, "exact", -1.0); }, Error::wrong_value),
          "is_intersecting: tol");
    check(test::raises([&] { Zonotope::stack({Z, Z}).is_intersecting(Z); }, Error::not_supported),
          "is_intersecting: batch");
    ChecksScope off(false);
    check(Z.is_intersecting(Z), "is_intersecting: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        intersecting_known(b);
        intersecting_property(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
