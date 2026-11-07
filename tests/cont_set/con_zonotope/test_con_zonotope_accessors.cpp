// test_con_zonotope_accessors - ConZonotope: the dimension and the generators
//
// CORA: contSet/@conZonotope/dim.m
// CORA: contSet/@conZonotope/generators.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const ConZonotope cz = test::hexagon();
        check(cz.dim() == 2 && Set(cz)->dim() == 2,
              b + ": the dimension is the rows of the center");
        check(close(cz.generators(), {1, 0, 1, 0, 1, 1}, 0) &&
                  cz.generators().shape() == Shape{2, 3},
              b + ": the generators");
    });
    return test::finish("con_zonotope accessors");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
