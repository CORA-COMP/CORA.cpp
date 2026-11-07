// test_zonotope_accessors - Zonotope: the dimension and the generators
//
// CORA: contSet/@zonotope/dim.m
// CORA: contSet/@zonotope/generators.m
// CORA authors: Mark Wetzlinger, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Zonotope Z(column({1.0, 2.0, 3.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}, {2.0, 2.0}}));
        check(Z.dim() == 3 && Set(Z)->dim() == 3, b + ": the dimension is the rows of the center");
        check(close(Z.generators(), {1, 0, 0, 1, 2, 2}, 0) && Z.generators().shape() == Shape{3, 2},
              b + ": the generators");
    });
    return test::finish("zonotope accessors");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
