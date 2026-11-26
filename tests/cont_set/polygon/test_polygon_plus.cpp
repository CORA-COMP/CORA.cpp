// test_polygon_plus - Polygon::plus: the Minkowski sum with a polygon or a point; minus
//
// CORA: contSet/@polygon/plus.m
// CORA: contSet/@polygon/minus.m
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
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void sums(const std::string &b) {
    const Polygon S = test::unit_square().plus(test::unit_square());
    check(S.num_pieces() == 1 && close(S.volume(), 4.0), b + ": two unit squares make [0, 2]^2");
    const Polygon shifted = test::l_shape().plus(vec({1.0, -1.0}));
    check(close(shifted.interval().inf(), {1.0, -1.0}), b + ": a point shifts");
    const Set back = shifted.minus(vec({1.0, -1.0}));
    check(close(back.as<Polygon>().interval().sup(), {2.0, 2.0}), b + ": minus shifts back");
    check(close((Set(test::unit_square()) + Set(test::unit_square())).as<Polygon>().volume(), 4.0),
          b + ": the operator +");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { sums(b); });
    check(test::raises([] { test::unit_square().plus(vec({1.0, 0.0, 0.0})); },
                       Error::dimension_mismatch),
          "plus: a point of three entries is refused");
    {
        ChecksScope off(false);
        check(close(test::unit_square().plus(test::unit_square()).volume(), 4.0),
              "plus: the same with the checks off");
    }
    return test::finish("polygon plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
