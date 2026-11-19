// test_ellipsoid_reduce - Ellipsoid reduce: an ellipsoid has no generators to reduce
//
// CORA: contSet/@ellipsoid/reduce.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The reduced ellipsoid is the same.
void reduce_known(const std::string &b) {
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(E.isequal(E.reduce(2).as<Ellipsoid>(), 0.0), b + ": reduce");
}

/// An order below 1 is refused; with the checks off a valid call works.
void checks_and_their_switch() {
    const Ellipsoid E(vec({1.0, 2.0}), Tensor({{4.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.reduce(0); }, Error::wrong_value), "reduce: order >= 1");
    ChecksScope off(false);
    check(E.reduce(1)->dim() == 2, "reduce: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { reduce_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
