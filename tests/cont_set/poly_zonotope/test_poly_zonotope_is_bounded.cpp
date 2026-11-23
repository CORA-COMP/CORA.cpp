// test_poly_zonotope_is_bounded - PolyZonotope is_bounded: a polynomial zonotope is bounded
//
// Every polynomial zonotope is: the factors range over [-1, 1].
//
// CORA: contSet/@polyZonotope/isBounded.m
// PROPERTY: contSet/@polyZonotope/isBounded.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(5);
        check(test::random_poly(rng, 3).is_bounded(), b + ": a random set is bounded");
        check(PolyZonotope::origin(2).is_bounded(), b + ": the origin is bounded");
    });
    ChecksScope off(false);
    check(PolyZonotope::origin(2).is_bounded(), "is_bounded: valid with the checks off");
    return test::finish("poly_zonotope is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
