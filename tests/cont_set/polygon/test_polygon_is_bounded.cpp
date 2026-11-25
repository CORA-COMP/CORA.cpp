// test_polygon_is_bounded - Polygon::is_bounded: always true
//
// CORA: contSet/@polygon/isBounded.m
// CORA authors: Tobias Ladner
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

void bounded() {
    check(test::l_shape().is_bounded() && Polygon::empty().is_bounded(), "is_bounded: always");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    bounded();
    {
        ChecksScope off(false);
        check(test::l_shape().is_bounded(), "is_bounded: the same with the checks off");
    }
    return test::finish("polygon is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
