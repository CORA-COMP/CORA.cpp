// test_polygon_isequal - Polygon::isequal: whether the pieces cover each other
//
// CORA: contSet/@polygon/isequal.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void equal() {
    check(test::l_shape().isequal(test::l_shape()), "isequal: a polygon is itself");
    check(!test::l_shape().isequal(test::unit_square()), "isequal: the L is not the square");
    const Polygon twice = test::unit_square().unite(test::boundary({{0, 0}, {1, 0}, {0, 1}}));
    check(twice.isequal(test::unit_square()), "isequal: a piece inside another changes nothing");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    equal();
    check(test::raises([] { test::l_shape().isequal(test::l_shape(), 0.5); }, Error::not_supported),
          "isequal: a tolerance is not supported");
    {
        ChecksScope off(false);
        check(test::l_shape().isequal(test::l_shape()), "isequal: the same with the checks off");
    }
    return test::finish("polygon isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
