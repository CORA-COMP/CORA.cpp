// test_polygon_representsa - Polygon::representsa: the empty set and polygon
//
// CORA: contSet/@polygon/representsa_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
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

void types() {
    check(Polygon::empty().representsa("emptySet") && !test::l_shape().representsa("emptySet") &&
              test::l_shape().representsa("polygon"),
          "representsa: emptySet and polygon");
    check(test::raises([] { test::l_shape().representsa("zonotope"); }, Error::not_supported),
          "representsa: another type is not supported");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    types();
    {
        ChecksScope off(false);
        check(Polygon::empty().representsa("emptySet"),
              "representsa: the same with the checks off");
    }
    return test::finish("polygon representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
