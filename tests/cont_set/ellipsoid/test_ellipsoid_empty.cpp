// test_ellipsoid_empty - Ellipsoid::empty: the empty ellipsoid of n dimensions
//
// CORA: contSet/@ellipsoid/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
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
    check(E.Q().shape() == Shape{0, 0} && E.q().shape() == Shape{3, 0},
          b + ": Q has no entries, q has the shape (n, 0)");
    check(E.dim() == 3 && Ellipsoid::empty().dim() == 0, b + ": the dimension is n");
    check(E.center().shape() == Shape{3, 0}, b + ": the center has no entries");
    check(!E.is_full_dim() && E.is_bounded(), b + ": empty: no interior, bounded");
    check(E.generators().shape() == Shape{3, 0}, b + ": empty: no generators");
    check(E.interval().is_empty(), b + ": empty: the empty box");
    // the single-argument form with an empty Q is the same kind of set
    check(Ellipsoid(Tensor::zeros({0, 0})).Q().numel() == 0, b + ": an empty Q alone");
}

/// A negative dimension is refused; with the checks off the call is the same.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid::empty(-1); }, Error::wrong_value),
          "empty: a negative dimension");
    const Ellipsoid on = Ellipsoid::empty(2);
    ChecksScope off(false);
    check(Ellipsoid::empty(2).q().shape() == on.q().shape(), "empty: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { empty_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
