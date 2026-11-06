// test_polytope_unite - Polytope::unite: the convex hull of the union
//
// CORA: contSet/@polytope/or.m
// PROPERTY: contSet/@polytope/or.m
//
// Property: the result contains both sets and has the vertices of their hull
//
// CORA authors: Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The hull of two squares and of a square and a point and a box.
void hulls(const std::string &b) {
    const Polytope far(Tensor({{2.0, 3.0, 3.0, 2.0}, {0.0, 0.0, 1.0, 1.0}}));
    const Polytope hull = test::square_v().unite(far);
    check(hull.contains(column({1.5, 0.5})) && hull.contains(column({3.0, 1.0})) &&
              !hull.contains(column({3.1, 0.5})),
          b + ": two squares");
    check(test::square_v().unite(column({3.0, 0.0})).contains(column({2.0, 0.0})), b + ": a point");
    check(test::square_v()
              .unite(Interval(column({2.0, 0.0}), column({3.0, 1.0})))
              .contains(column({2.5, 0.5})),
          b + ": a box");
    // the same as convex_hull
    check(hull.isequal(test::square_v().convex_hull(far)), b + ": it is convex_hull");
}

/// The hull contains both operands.
void properties(const std::string &b) {
    Rng rng(4);
    bool contained = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P1(test::sphere_points(rng, n, 5)),
            P2(test::sphere_points(rng, n, 5, 0.5, 1.0));
        const Polytope hull = P1.unite(P2);
        contained = contained && hull.contains(P1.V(), "exact", 1e-8) &&
                    hull.contains(P2.V(), "exact", 1e-8);
    }
    check(contained, b + ": the hull contains both sets");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        hulls(b);
        properties(b);
    });
    check(test::raises([] { test::square_v().unite(Polytope::inf(3)); }, Error::dimension_mismatch),
          "unite: dimensions");
    ChecksScope off(false);
    check(test::square_v().unite(test::square_v()).dim() == 2,
          "unite: the same with the checks off");
    return test::finish("polytope unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
