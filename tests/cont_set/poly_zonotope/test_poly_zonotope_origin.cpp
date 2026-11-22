// test_poly_zonotope_origin - PolyZonotope origin: the set {0}
//
// CORA: contSet/@polyZonotope/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       22-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the center is 0 and there are no generators
        const PolyZonotope P = PolyZonotope::origin(3);
        check(P.dim() == 3 && close(P.center(), {0.0, 0.0, 0.0}, 0), b + ": the origin of R^3");
        check(P.num_dependent() == 0 && P.independent_generators().size(0) == 0,
              b + ": no generators");
    });

    // the dimension is positive
    check(test::raises([] { static_cast<void>(PolyZonotope::origin(0)); }, Error::wrong_value),
          "origin: the dimension");
    ChecksScope off(false);
    check(PolyZonotope::origin(2).dim() == 2, "origin: valid with the checks off");
    return test::finish("poly_zonotope origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
