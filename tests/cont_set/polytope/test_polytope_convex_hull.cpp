// test_polytope_convex_hull - Polytope::convex_hull: the hull with a polytope, a box or a point
//
// CORA: contSet/@polytope/convHull_.m
// PROPERTY: contSet/@polytope/convHull_.m
//
// Property: the hull holds both sets and its vertices are vertices of one of them
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two squares, a box, a point, the special sets.
void hulls(const std::string &b) {
    const Polytope S1(Tensor({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}));
    const Polytope S2(Tensor({{2.0, 0.0}, {3.0, 0.0}, {3.0, 1.0}, {2.0, 1.0}}));
    const Polytope H = S1.convex_hull(S2);
    check(H.is_v_rep() && H.V().shape() == Shape{4, 2}, b + ": two squares: 4 extreme vertices");
    check(H.contains(vec({1.5, 0.5})) && !H.contains(vec({1.5, 1.5})), b + ": the points");
    const Polytope box = S1.convex_hull(Interval(vec({2.0, 0.0}), vec({3.0, 1.0})));
    check(box.V().shape() == Shape{4, 2} && box.contains(vec({2.5, 0.5})), b + ": a box");
    const Polytope pt = S1.convex_hull(vec({2.0, 0.5}));
    check(pt.V().shape() == Shape{5, 2} && pt.contains(vec({1.5, 0.5})), b + ": a point");
    check(S1.convex_hull(vec({0.5, 0.5})).V().shape() == Shape{4, 2}, b + ": an inner point goes");
    // halfspace sets are hulled through their vertices
    const Polytope T(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), vec({0.0, 0.0, 1.0}));
    const Polytope TS = T.convex_hull(S2);
    check(TS.contains(vec({0.0, 0.0})) && TS.contains(vec({3.0, 1.0})) &&
              TS.contains(vec({1.0, 0.5})),
          b + ": a halfspace polytope");
    // the hull with the empty set is the other set, with R^n it is R^n
    check(close(S1.convex_hull(Polytope::empty(2)).V(), S1.V(), 0), b + ": with the empty set");
    check(close(Polytope::empty(2).convex_hull(S1).V(), S1.V(), 0), b + ": the empty set first");
    check(Polytope::empty(2).convex_hull(Polytope::empty(2)).V().shape() == Shape{0, 2},
          b + ": two empty sets");
    check(S1.convex_hull(Polytope::inf(2)).A().shape() == Shape{0, 2}, b + ": with R^n");
    check(S1.convex_hull(Interval::empty(2)).V().shape() == Shape{4, 2},
          b + ": with the empty box");
    check(S1.convex_hull(Interval::unbounded(2)).A().shape() == Shape{0, 2},
          b + ": with the whole box");
    // in 1D the hull is the interval between the extremes
    check(close(Polytope(Tensor({{0.0}, {1.0}})).convex_hull(Polytope(Tensor({{3.0}, {4.0}}))).V(),
                {0, 4}, 0),
          b + ": 1D");
}

/// What is refused; checks off.
void refusals() {
    const Polytope S(Tensor({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}}));
    check(test::raises([&] { S.convex_hull(Polytope::inf(3)); }, Error::dimension_mismatch),
          "convex_hull: dimensions");
    check(test::raises([&] { S.convex_hull(Tensor::zeros({3, 1})); }, Error::dimension_mismatch),
          "convex_hull: the point has the dimension");
    check(test::raises(
              [&] {
                  S.convex_hull(Interval(vec({0.0, 0.0}),
                                         vec({1.0, std::numeric_limits<double>::infinity()})));
              },
              Error::not_supported),
          "convex_hull: an unbounded box");
    ChecksScope off(false);
    check(S.convex_hull(vec({3.0, 1.0})).V().shape() == Shape{4, 2},
          "convex_hull: the same with the checks off");
}

/// The hull holds both sets and its vertices are vertices of one of them; the mix of a point of
/// each is inside.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3}) {
        const Tensor first = test::sphere_points(rng, n, n + 3),
                     second = test::sphere_points(rng, n, n + 3, 0.5, 2.0);
        const Polytope hull = Polytope(first).convex_hull(Polytope(second));
        check(hull.contains(first, "exact", 1e-8) && hull.contains(second, "exact", 1e-8),
              b + ": it holds both sets, " + std::to_string(n) + "D");
        const Tensor mix = first.take({0}, -2) * 0.3 + second.take({1}, -2) * 0.7;
        check(hull.contains(mix, "exact", 1e-8),
              b + ": a mix of two points, " + std::to_string(n) + "D");
        const Tensor all = Tensor::concat({first, second}, -2);
        check(hull.V().size(-2) <= all.shape()[0],
              b + ": no new vertices, " + std::to_string(n) + "D");
        check(!hull.contains(second.take({0}, -2) * 3.0, "exact", 1e-8),
              b + ": a far point, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        hulls(b);
        properties(b);
    });
    refusals();
    return test::finish("polytope convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
