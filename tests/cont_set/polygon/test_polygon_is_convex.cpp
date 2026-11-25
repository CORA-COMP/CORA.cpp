// test_polygon_is_convex - Polygon::is_convex: one piece holds all
//
// CORA: contSet/@polygon/isConvex.m
// CORA authors: Niklas Kochdumper
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

void convexity() {
    check(test::unit_square().is_convex() && Polygon::empty().is_convex(),
          "is_convex: a square, nothing");
    check(!test::l_shape().is_convex(), "is_convex: the L is not");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    convexity();
    {
        ChecksScope off(false);
        check(test::unit_square().is_convex(), "is_convex: the same with the checks off");
    }
    return test::finish("polygon is_convex");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
