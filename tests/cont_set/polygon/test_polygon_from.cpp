// test_polygon_from - Polygon::from: a two-dimensional set as a polygon
//
// CORA: contSet/@polygon/polygon.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void sets(const std::string &b) {
    const Polygon box = Polygon::from(Interval(vec({0.0, 0.0}), vec({2.0, 1.0})));
    check(box.num_pieces() == 1 && close(box.volume(), 2.0), b + ": a box, area 2");
    const Zonotope Z(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {1.0, 1.0}}));
    check(close(Polygon::from(Z).volume(), 4.0), b + ": a parallelogram, area 4");
    check(Polygon::from(test::l_shape()).num_pieces() == test::l_shape().num_pieces(),
          b + ": a polygon is itself");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { sets(b); });
    check(test::raises([] { Polygon::from(Interval(vec({0.0, 0.0, 0.0}), vec({1.0, 1.0, 1.0}))); },
                       Error::dimension_mismatch),
          "from: a set of three dimensions is refused");
    {
        ChecksScope off(false);
        check(close(Polygon::from(Interval(vec({0.0, 0.0}), vec({1.0, 1.0}))).volume(), 1.0),
              "from: the same with the checks off");
    }
    return test::finish("polygon from");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
