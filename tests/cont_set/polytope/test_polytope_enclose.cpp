// test_polytope_enclose - Polytope::enclose: the convex hull of a set and another
//
// CORA: contSet/@polytope/enclose.m
// PROPERTY: contSet/@polytope/enclose.m
//
// Property: the result is the convex hull of the two sets
//
// CORA authors: Matthias Althoff, Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// enclose is the hull; the Set interface dispatches to it.
void enclosing(const std::string &b) {
    const Polytope S1(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    const Polytope S2(Tensor({{2.0, 3.0, 3.0, 2.0}, {0.0, 0.0, 1.0, 1.0}}));
    const Polytope E = S1.enclose(S2);
    check(close(E.V(), S1.convex_hull(S2).V(), 0), b + ": the convex hull");
    check(E.contains(column({1.5, 0.5})), b + ": the gap is inside");
    const ContSet &S = S1;
    check(S.enclose(S2).is<Polytope>(), b + ": the set interface encloses a set");
}

/// The form with a map and a sum is refused; checks off.
void refusals() {
    const Polytope S(Tensor({{0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}));
    check(test::raises([&] { S.enclose(Tensor::eye(2), S); }, Error::not_supported),
          "enclose: M P and the sum are not ported");
    check(test::raises([&] { S.enclose(Polytope::inf(3)); }, Error::dimension_mismatch),
          "enclose: dimensions");
    ChecksScope off(false);
    check(S.enclose(S).V().shape() == Shape{2, 3}, "enclose: the same with the checks off");
}

/// The enclosure of two random vertex sets is their convex hull.
void properties(const std::string &b) {
    Rng rng(4);
    const Tensor first = test::sphere_points(rng, 3, 6),
                 second = test::sphere_points(rng, 3, 6, 0.5, 2.0);
    const Polytope P1(first), P2(second);
    check(test::same_columns(P1.enclose(P2).V(), P1.convex_hull(P2).V()),
          b + ": the same vertices");
    check(P1.enclose(P2).contains(first, "exact", 1e-8) &&
              P1.enclose(P2).contains(second, "exact", 1e-8),
          b + ": it holds both sets");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclosing(b);
        properties(b);
    });
    refusals();
    return test::finish("polytope enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
