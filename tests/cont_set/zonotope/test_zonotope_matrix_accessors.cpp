// test_zonotope_matrix_accessors - matrix zonotope: the shape and the number of generators
//
// CORA: matrixSet/@matZonotope/numgens.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/matrix_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        check(test::example().nr_generators() == 2, b + ": two generators");
        check(test::point_set(test::matrix({1, 2, 3, 4}, 2, 2)).nr_generators() == 0,
              b + ": a point has none");
    });
    return test::finish("mat_zonotope accessors");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
