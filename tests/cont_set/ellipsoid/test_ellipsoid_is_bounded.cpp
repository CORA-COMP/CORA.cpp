// test_ellipsoid_is_bounded - Ellipsoid::is_bounded: always true
//
// CORA: contSet/@ellipsoid/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        check(Ellipsoid(Tensor({{1.0}})).is_bounded(), b + ": a ball");
        check(Ellipsoid::empty(2).is_bounded(), b + ": the empty ellipsoid");
        const Set S = Ellipsoid(Tensor({{1.0, 0.0}, {0.0, 0.0}}));
        check(S->is_bounded(), b + ": through the interface of every set");
    });
    // no input to get wrong; the same with the checks off
    ChecksScope off(false);
    check(Ellipsoid(Tensor({{1.0}})).is_bounded(), "is_bounded: the same with the checks off");
    return test::finish("ellipsoid is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
