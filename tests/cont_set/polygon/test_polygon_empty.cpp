// test_polygon_empty - Polygon::empty: the polygon without a piece
//
// CORA: contSet/@polygon/empty.m
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

void the_empty_polygon() {
    const Polygon E = Polygon::empty();
    check(E.num_pieces() == 0 && E.representsa("emptySet") && E.volume() == 0,
          "empty: no piece, no area");
    check(test::raises([] { Polygon::empty(3); }, Error::wrong_value), "empty: n = 3 is refused");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    the_empty_polygon();
    {
        ChecksScope off(false);
        check(Polygon::empty().num_pieces() == 0, "empty: the same with the checks off");
    }
    return test::finish("polygon empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
