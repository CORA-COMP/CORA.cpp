// test_polygon_project - Polygon::project: the swap of the coordinates, or one
//
// CORA: contSet/@polygon/project.m
// CORA authors: Tobias Ladner
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

void projections(const std::string &b) {
    const Set swapped = test::l_shape().project({1, 0});
    check(close(swapped.as<Polygon>().volume(), 3.0), b + ": the swap keeps the area");
    const Set x = test::l_shape().project({0});
    check(close(x->interval().sup(), {2.0}), b + ": one coordinate is its interval");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { projections(b); });
    check(test::raises([] { test::l_shape().project({0, 2}); }, Error::wrong_value),
          "project: a coordinate out of range is refused");
    {
        ChecksScope off(false);
        check(close(test::l_shape().project({1, 0}).as<Polygon>().volume(), 3.0),
              "project: the same with the checks off");
    }
    return test::finish("polygon project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
