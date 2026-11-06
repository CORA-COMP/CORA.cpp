// test_polytope_representsa - Polytope::representsa: whether the set is also a set of another type
//
// CORA: contSet/@polytope/representsa_.m
// CORA: contSet/@polytope/isemptyobject.m
// CORA: contSet/@contSet/representsa_emptyObject.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
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

/// The types with a test of their own: origin, point, emptySet, fullspace.
void points_and_extremes(const std::string &b) {
    check(Polytope::origin(2).representsa("origin") && !test::square_h().representsa("origin"),
          b + ": origin");
    check(Polytope(column({1.0, 2.0})).representsa("point") &&
              !test::square_v().representsa("point"),
          b + ": point");
    check(!Polytope::empty(2).representsa("point"), b + ": the empty set is no point");
    check(Polytope::empty(2).representsa("emptySet") && !test::square_h().representsa("emptySet"),
          b + ": emptySet");
    check(Polytope::inf(2).representsa("fullspace") && !test::square_h().representsa("fullspace"),
          b + ": fullspace");
    // a point in one dimension: one vertex
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({-2.0, 2.0})).representsa("point"),
          b + ": 1D point");
}

/// Boxes, halfspaces, hyperplanes and the types that ask for boundedness.
void shapes(const std::string &b) {
    check(test::square_h().representsa("interval") && !test::triangle_h().representsa("interval"),
          b + ": interval");
    const Polytope halfspace(Tensor({{1.0, 1.0}}), column({1.0}));
    const Polytope parallel(Tensor({{1.0, 0.0}, {2.0, 0.0}}), column({1.0, 3.0}));
    check(halfspace.representsa("halfspace") && parallel.representsa("halfspace") &&
              !test::square_h().representsa("halfspace"),
          b + ": halfspace");
    const Polytope plane(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 1.0}}),
                         column({1.0}));
    check(plane.representsa("hyperplane") && !halfspace.representsa("hyperplane"),
          b + ": hyperplane");
    check(plane.representsa("conHyperplane") && !test::square_h().representsa("conHyperplane"),
          b + ": conHyperplane");
    // the bounded sets are constrained zonotopes and bundles
    for (const char *type : {"conZonotope", "zonoBundle", "polyZonotope", "conPolyZono"})
        check(test::square_h().representsa(type) && !halfspace.representsa(type),
              b + ": bounded: " + std::string(type));
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0})).representsa("capsule") &&
              !test::square_h().representsa("capsule"),
          b + ": capsule");
    check(Polytope(column({1.0, 1.0})).representsa("ellipsoid") &&
              !test::square_h().representsa("ellipsoid"),
          b + ": ellipsoid");
    check(test::square_h().representsa("polytope") && test::square_h().representsa("convexSet") &&
              test::square_h().representsa("levelSet") &&
              !test::square_h().representsa("probZonotope"),
          b + ": the types that are always or never");
}

/// The tables for a set without constraints (CORA's empty object).
void empty_object(const std::string &b) {
    const Polytope R = Polytope::inf(2);
    check(R.representsa("polytope") && R.representsa("hyperplane") && R.representsa("interval"),
          b + ": R^n: always");
    check(!R.representsa("origin") && !R.representsa("point") && !R.representsa("zonoBundle"),
          b + ": R^n: never");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_and_extremes(b);
        shapes(b);
        empty_object(b);
    });
    check(test::raises([] { test::square_h().representsa("zonotope"); }, Error::not_supported),
          "representsa: zonotope is not decided");
    check(test::raises([] { test::square_h().representsa("ball"); }, Error::wrong_value),
          "representsa: an unknown type");
    ChecksScope off(false);
    check(test::square_h().representsa("interval"), "representsa: the same with the checks off");
    return test::finish("polytope representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
