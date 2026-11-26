// test_polygon_interval - Polygon::interval: the box of the pieces
//
// CORA: contSet/@polygon/interval.m
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

void boxes(const std::string &b) {
    const Interval I = test::l_shape().interval();
    check(close(I.inf(), {0.0, 0.0}) && close(I.sup(), {2.0, 2.0}), b + ": the box of the L");
    check(Polygon::empty().interval().is_empty(), b + ": no piece, the empty box");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { boxes(b); });
    {
        ChecksScope off(false);
        check(close(test::unit_square().interval().sup(), {1.0, 1.0}),
              "interval: the same with the checks off");
    }
    return test::finish("polygon interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
