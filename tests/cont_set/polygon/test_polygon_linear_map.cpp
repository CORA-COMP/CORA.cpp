// test_polygon_linear_map - Polygon::linear_map: every piece mapped by a 2 x 2 matrix; times
//
// CORA: contSet/@polygon/mtimes.m
// CORA: contSet/@polygon/times.m
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

void maps(const std::string &b) {
    const Tensor M({{2.0, 1.0}, {0.0, 3.0}});
    const Set R = test::l_shape().linear_map(M);
    check(close(R.as<Polygon>().volume(), 18.0), b + ": the area scales by det M = 6");
    const Set T = test::l_shape().times(vec({2.0, 0.5}));
    check(close(T.as<Polygon>().volume(), 3.0), b + ": times by (2, 0.5) keeps the area");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { maps(b); });
    check(test::raises(
              [] { test::l_shape().linear_map(Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}})); },
              Error::dimension_mismatch),
          "linear_map: a matrix of three rows is refused");
    {
        ChecksScope off(false);
        check(
            close(
                test::l_shape().linear_map(Tensor({{1.0, 0.0}, {0.0, 1.0}})).as<Polygon>().volume(),
                3.0),
            "linear_map: the same with the checks off");
    }
    return test::finish("polygon linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
