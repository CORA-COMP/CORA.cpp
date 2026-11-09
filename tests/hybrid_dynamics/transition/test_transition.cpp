// test_transition - Transition and Location: their constructors and checks
//
// A box guard becomes the polytope of its sides; a location holds its flow, invariant and jumps,
// and c defaults to zero.
//
// CORA: hybridDynamics/@transition/transition.m
// CORA: hybridDynamics/@location/location.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/location/location.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The guard, the reset and the target; the parts of a location.
void construction() {
    const Interval box(column({0.0, -1.0}), column({1.0, 1.0}));
    const Transition t(box, LinearReset(Tensor::eye(2)), 1);
    check(t.target() == 1 && t.guard().dim() == 2, "transition: target and guard");
    check(t.guard().contains(column({0.5, 0.0})) && !t.guard().contains(column({2.0, 0.0})),
          "transition: the box as a polytope");
    // a location without B, and one with B and c
    const Polytope inv(Tensor({{1.0, 0.0}}), column({0.0}));
    const Location loc("free", inv, {t}, LinearSys(Tensor({{0.0, 1.0}, {0.0, 0.0}})));
    check(loc.name() == "free" && loc.dim() == 2 && loc.transitions().size() == 1 &&
              loc.input_dim() == 0 && close(loc.c(), {0, 0}, 0),
          "location: its parts, c = 0 by default");
    const Location with_c("fall", inv, {}, LinearSys(Tensor::zeros({2, 2}), Tensor::zeros({2, 1})),
                          column({0.0, -9.81}));
    check(with_c.input_dim() == 1 && close(with_c.c(), {0, -9.81}), "location: B and c");
}

/// Wrong dimensions and targets are refused; with the checks off construction is the same.
void checks_and_their_switch() {
    const LinearReset id(Tensor::eye(2));
    const Polytope guard(Tensor({{1.0, 0.0}}), column({0.0}));
    check(test::raises([&] { Transition(guard, LinearReset(Tensor::eye(3)), 0); },
                       Error::wrong_input_in_constructor),
          "transition: the guard and the reset differ in dimension");
    check(test::raises([&] { Transition(guard, id, -1); }, Error::wrong_input_in_constructor),
          "transition: a negative target");
    // a location's invariant and c have its dimension
    const LinearSys flow(Tensor::zeros({2, 2}));
    check(test::raises([&] { Location("a", Polytope(Tensor({{1.0}}), column({0.0})), {}, flow); },
                       Error::wrong_input_in_constructor),
          "location: the invariant of another dimension");
    check(test::raises([&] { Location("a", guard, {}, flow, column({1.0})); },
                       Error::wrong_input_in_constructor),
          "location: c of another dimension");
    check(test::raises(
              [&] {
                  Location("a", guard, {},
                           LinearSys(Tensor::zeros({2, 2}), Tensor::zeros({2, 1}), Tensor::eye(2)));
              },
              Error::wrong_input_in_constructor),
          "location: an output map C");
    ChecksScope off(false);
    check(Transition(guard, id, 0).target() == 0, "transition: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    construction();
    checks_and_their_switch();
    return test::finish("transition and location");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
