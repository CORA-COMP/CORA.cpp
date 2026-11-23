// test_poly_zonotope_empty - PolyZonotope empty: the empty polynomial zonotope is refused
//
// A polynomial zonotope has no empty member here: empty(n) is Error::not_supported, a negative
// dimension Error::wrong_value.
//
// CORA: contSet/@polyZonotope/empty.m
// CORA authors: Mark Wetzlinger
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
    check(test::raises([] { PolyZonotope::empty(2); }, Error::not_supported),
          "empty: refused as not supported");
    check(test::raises([] { PolyZonotope::empty(-1); }, Error::wrong_value),
          "empty: a negative dimension");
    ChecksScope off(false);
    check(test::raises([] { PolyZonotope::empty(0); }, Error::not_supported),
          "empty: still refused with the checks off");
    return test::finish("poly_zonotope empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
