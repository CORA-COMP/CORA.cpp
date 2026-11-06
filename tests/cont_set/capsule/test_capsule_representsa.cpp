// test_capsule_representsa - Capsule representsa: the types CORA answers without a conversion
//
// CORA: contSet/@capsule/representsa_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed for a point, a segment, a ball, a proper capsule, a line in 1D and the empty one.
void representsa_known(const std::string &b) {
    const Capsule point(column({0.0, 0.0})), shifted(column({1.0, 0.0})),
        segment(column({0.0, 0.0}), column({1.0, 0.0}), 0.0);
    const Capsule ball(column({0.0, 0.0}), column({0.0, 0.0}), 1.0),
        full(column({0.0, 0.0}), column({1.0, 0.0}), 1.0);
    // points and segments first
    check(point.representsa("origin") && point.representsa("point"), b + ": the origin is a point");
    check(!shifted.representsa("origin") && shifted.representsa("point"), b + ": a shifted point");
    check(!segment.representsa("point") && segment.representsa("zonotope") &&
              segment.representsa("polytope"),
          b + ": a segment");
    check(segment.representsa("interval") &&
              !Capsule(column({0.0, 0.0}), column({1.0, 1.0}), 0.0).representsa("interval"),
          b + ": interval needs an axis");
    // balls, capsules and the empty set
    check(ball.representsa("ellipsoid") && !full.representsa("ellipsoid") &&
              !ball.representsa("zonotope"),
          b + ": a ball");
    check(full.representsa("capsule") && full.representsa("convexSet") &&
              !full.representsa("halfspace"),
          b + ": a capsule");
    check(!full.representsa("emptySet") && Capsule::empty(2).representsa("emptySet"),
          b + ": emptySet");
    check(Capsule(column({1.0}), column({2.0}), 1.0).representsa("zonotope") &&
              full.representsa("hyperplane") == false,
          b + ": 1D");
    check(Capsule(column({1.0}), column({2.0}), 1.0).representsa("hyperplane"),
          b + ": hyperplane in 1D");
    check(Capsule::empty(2).representsa("capsule") && !Capsule::empty(2).representsa("point"),
          b + ": the empty capsule");
    check(point.representsa("origin", 0.0) &&
              Capsule(column({1e-9, 0.0})).representsa("origin", 1e-6) &&
              !Capsule(column({1e-3, 0.0})).representsa("origin", 1e-6),
          b + ": the tolerance");
}

/// An unknown type, a type CORA refuses, a batch, a negative tolerance; checks off works.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.representsa("sphere"); }, Error::wrong_value),
          "representsa: unknown type");
    check(test::raises([&] { C.representsa("levelSet"); }, Error::not_supported),
          "representsa: levelSet");
    check(test::raises([&] { C.representsa("point", -1.0); }, Error::wrong_value),
          "representsa: tol");
    const Tensor z = Tensor::zeros({2, 2, 1});
    check(test::raises([&] { Capsule(z, z, 0.0).representsa("point"); }, Error::not_supported),
          "representsa: batch");
    ChecksScope off(false);
    check(!C.representsa("point"), "representsa: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { representsa_known(b); });
    checks_and_their_switch();
    return test::finish("capsule representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
