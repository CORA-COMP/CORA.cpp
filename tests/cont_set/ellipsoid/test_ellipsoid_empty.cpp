// test_ellipsoid_empty - Ellipsoid::empty: the empty ellipsoid of n dimensions
//
// CORA: contSet/@ellipsoid/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shapes of the empty ellipsoid and the queries on it.
void empty_known(const std::string &b) {
    const Ellipsoid E = Ellipsoid::empty(3);
    check(E.is_empty() && E.empty_mask().all() && E.Q().shape() == Shape{3, 3} &&
              E.q().shape() == Shape{3},
          b + ": flagged in the mask, the shapes keep n");
    check(E.dim() == 3 && Ellipsoid::empty().dim() == 0, b + ": the dimension is n");
    check(E.center().shape() == Shape{3, 1} && std::isnan(E.center().data()[0]),
          b + ": the center is NaN");
    check(!E.is_full_dim() && E.is_bounded(), b + ": empty: no interior, bounded");
    check(E.generators().shape() == Shape{0, 3}, b + ": empty: no generators");
    check(E.interval().is_empty(), b + ": empty: the empty box");
    // a set of data is not empty: the shape matrix 0 is a point
    check(!Ellipsoid(Tensor::zeros({3, 3})).is_empty() && !Ellipsoid::origin(3).is_empty(),
          b + ": a point is not empty");
}

/// A negative dimension is refused; with the checks off the call is the same.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid::empty(-1); }, Error::wrong_value),
          "empty: a negative dimension");
    const Ellipsoid on = Ellipsoid::empty(2);
    ChecksScope off(false);
    check(Ellipsoid::empty(2).q().shape() == on.q().shape() && Ellipsoid::empty(2).is_empty(),
          "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { empty_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
