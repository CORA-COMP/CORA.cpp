// test_zonotope_convex_hull - zonotope convex_hull: an enclosure of the hull with another set
//
// CORA: contSet/@zonotope/convHull_.m
// PROPERTY: contSet/@zonotope/convHull_.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A zonotope, a box and a point: the hull encloses each, and equals enclose of the converted set.
void convex_hull_with_each_kind_of_set(const std::string &b) {
    cora::Rng rng(19);
    const int n = 3;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 4),
                   Z2 = Zonotope::generate_random(rng, n, std::nullopt, 4);
    const Interval I = Interval::generate_random(rng, n);
    const Tensor &p = Z2.center();
    check(close(Z.convex_hull(Z2).generators(), Z.enclose(Z2).generators(), 0),
          b + ": zonotope: enclose");
    const Zonotope with_box = Z.convex_hull(I), with_point = Z.convex_hull(p);
    check(with_point.generators().shape().back() == 5,
          b + ": a point has no generators: the hull has m + 1 generators");
    for (int k = 0; k < 40; ++k) {
        const std::vector<double> d = test::random_direction(rng, n);
        check(test::support(with_box, d) >=
                  std::max(test::support(Z, d), test::support(I, d)) - 1e-9,
              b + ": box");
        // the support along each direction
        check(test::support(with_point, d) >=
                  std::max(test::support(Z, d),
                           d[0] * p.data()[0] + d[1] * p.data()[1] + d[2] * p.data()[2]) -
                      1e-9,
              b + ": point");
    }
    const Set via_sets = convex_hull(Z, Z2);
    check(via_sets.is<Zonotope>() && via_sets.as<Zonotope>().dim() == n,
          b + ": through the registry");
    check(convex_hull(Z, I).is<Zonotope>(), b + ": a box through the registry");
}

/// Operands of another dimension, and a point that is not a column, are refused.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {0.0}})),
        Z1(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Z.convex_hull(Z1); }, Error::dimension_mismatch),
          "convex_hull: dim 1");
    check(test::raises([&] { Z.convex_hull(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "convex_hull: a box of dimension 1");
    check(test::raises([&] { Z.convex_hull(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "convex_hull: a point of dimension 3");
    const Tensor on = Z.convex_hull(Z).generators();
    ChecksScope off(false);
    check(test::close(Z.convex_hull(Z).generators(), on, 0),
          "convex_hull: same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { convex_hull_with_each_kind_of_set(b); });
    checks_and_their_switch();
    return test::finish("zonotope convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
