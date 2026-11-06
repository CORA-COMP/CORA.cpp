// test_ellipsoid_isequal - Ellipsoid isequal: the same Q and q up to a tolerance
//
// CORA: contSet/@ellipsoid/isequal.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Equal, a field off, the tolerance that closes the gap, empty ellipsoids, another dimension.
void isequal_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(E.isequal(Ellipsoid(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}))),
          b + ": equal");
    check(E.isequal(E, 0.0), b + ": equal with tol = 0");
    check(!E.isequal(Ellipsoid(Tensor({{4.0, 0.0}, {0.0, 1.1}}), column({1.0, 2.0}))), b + ": Q");
    check(!E.isequal(Ellipsoid(E.Q(), column({1.0, 2.1}))), b + ": q");
    check(E.isequal(Ellipsoid(E.Q(), column({1.0, 2.001})), 1e-2),
          b + ": a tolerance closes a small gap");
    check(Ellipsoid::empty(2).isequal(Ellipsoid::empty(2)) && !Ellipsoid::empty(2).isequal(E) &&
              !E.isequal(Ellipsoid::empty(2)),
          b + ": empty");
    check(!E.isequal(Ellipsoid(Tensor::eye(3))), b + ": dimension");
    check(Set(E) == Set(E) && Set(E) != Set(Ellipsoid(Tensor::eye(2))), b + ": operator==");
}

/// A negative tolerance is refused; with the checks off the answer is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.isequal(E, -1.0); }, Error::wrong_value),
          "isequal: tol is nonnegative");
    ChecksScope off(false);
    check(E.isequal(E), "isequal: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { isequal_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
