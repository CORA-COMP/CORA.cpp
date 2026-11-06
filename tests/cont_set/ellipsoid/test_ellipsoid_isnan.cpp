// test_ellipsoid_isnan - Ellipsoid isnan and dim: no NaN, and the dimension from the center
//
// CORA: contSet/@ellipsoid/isnan.m
// CORA: contSet/@ellipsoid/dim.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// isnan is false; dim is the number of rows of the center, also for the empty ellipsoid.
void isnan_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(!E.isnan() && !Ellipsoid::empty(2).isnan(), b + ": isnan");
    check(E.dim() == 2 && Ellipsoid::empty(3).dim() == 3 && Ellipsoid(Tensor::eye(4)).dim() == 4,
          b + ": dim");
}

/// The call works with the checks off too.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    ChecksScope off(false);
    check(!E.isnan() && E.dim() == 2, "isnan: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { isnan_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid isnan");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
