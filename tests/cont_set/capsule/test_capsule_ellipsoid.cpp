// test_capsule_ellipsoid - Capsule ellipsoid: a ball is the ellipsoid r^2 I
//
// CORA: contSet/@capsule/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/rng.h"

#include <cmath>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the ball, the empty capsule; a generator has no exact ellipsoid.
void ellipsoid_known(const std::string &b) {
    const Capsule ball(column({1.0, 2.0}), column({0.0, 0.0}), 3.0);
    const Ellipsoid E = ball.ellipsoid();
    check(close(E.Q(), {9, 0, 0, 9}, 0) && close(E.q(), {1, 2}, 0), b + ": r^2 I at c");
    check(close(ball.ellipsoid("outer").Q(), E.Q(), 0), b + ": a ball ignores the method");
    check(Capsule::empty(3).ellipsoid().dim() == 3 &&
              Capsule::empty(3).ellipsoid().Q().numel() == 0,
          b + ": empty");
}

/// Refusals: a generator is no exact ellipsoid, the unions need an SDP, a bad method, a batch.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.ellipsoid(); }, Error::no_exact_alg),
          "ellipsoid: exact with a generator");
    check(test::raises([&] { C.ellipsoid("outer"); }, Error::not_supported), "ellipsoid: outer");
    check(test::raises([&] { C.ellipsoid("middle"); }, Error::wrong_value), "ellipsoid: method");
    const Capsule ball(column({1.0, 2.0}), column({0.0, 0.0}), 3.0);
    ChecksScope off(false);
    check(ball.ellipsoid().dim() == 2, "ellipsoid: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { ellipsoid_known(b); });
    checks_and_their_switch();
    return test::finish("capsule ellipsoid");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
