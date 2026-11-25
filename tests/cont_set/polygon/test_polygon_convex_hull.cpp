// test_polygon_convex_hull - Polygon::convex_hull: the hull of the pieces, a vertex polytope
//
// CORA: contSet/@polygon/convHull_.m
// CORA: contSet/@polygon/polytope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void hulls(const std::string &b) {
    const Polytope H = test::l_shape().convex_hull();
    check(H.is_v_rep() && close(H.volume(), 3.5), b + ": the hull of the L has area 3.5");
    check(close(test::l_shape().polytope().volume(), 3.5), b + ": polytope is the hull");
    check(Polygon::empty().convex_hull().representsa("emptySet"), b + ": no piece, no hull");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { hulls(b); });
    {
        ChecksScope off(false);
        check(close(test::l_shape().convex_hull().volume(), 3.5),
              "convex_hull: the same with the checks off");
    }
    return test::finish("polygon convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
