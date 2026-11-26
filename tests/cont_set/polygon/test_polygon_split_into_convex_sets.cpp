// test_polygon_split_into_convex_sets - Polygon::split_into_convex_sets: the stored convex pieces
//
// CORA: contSet/@polygon/splitIntoConvexSets.m
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

void split() {
    const std::vector<Polytope> pieces = test::l_shape().split_into_convex_sets();
    double area = 0;
    for (const Polytope &P : pieces) area += P.volume();
    check(pieces.size() >= 2 && close(area, 3.0), "split_into_convex_sets: the L's pieces, area 3");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    split();
    {
        ChecksScope off(false);
        check(test::unit_square().split_into_convex_sets().size() == 1,
              "split_into_convex_sets: the same with the checks off");
    }
    return test::finish("polygon split_into_convex_sets");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
