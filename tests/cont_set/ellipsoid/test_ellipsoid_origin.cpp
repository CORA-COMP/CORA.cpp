// test_ellipsoid_origin - Ellipsoid origin: the point ellipsoid at 0
//
// CORA: contSet/@ellipsoid/origin.m
// CORA authors: Mark Wetzlinger
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

/// Q = 0 and q = 0 in n dimensions.
void origin_known(const std::string &b) {
    const Ellipsoid O = Ellipsoid::origin(3);
    check(O.dim() == 3 && close(O.Q(), Tensor::zeros({3, 3}), 0) &&
              close(O.q(), Tensor::zeros({3, 1}), 0),
          b + ": origin");
}

/// A dimension below 1 is refused; with the checks off a valid call works.
void checks_and_their_switch() {
    check(test::raises([] { Ellipsoid::origin(0); }, Error::wrong_value), "origin: n >= 1");
    ChecksScope off(false);
    check(Ellipsoid::origin(2).dim() == 2, "origin: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { origin_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
