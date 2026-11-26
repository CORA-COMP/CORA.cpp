// test_polygon_linear_combination - Polygon::linear_combination: the hulls of every pair of pieces
//
// CORA: contSet/@polygon/linComb.m
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

void combinations(const std::string &b) {
    const Polygon a = test::unit_square(), c = test::boundary({{3, 0}, {4, 0}, {4, 1}, {3, 1}});
    const Polygon L = a.linear_combination(c);
    check(L.num_pieces() == 1 && close(L.volume(), 4.0), b + ": the hull of two squares");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { combinations(b); });
    // an invalid operand is refused before the operation
    check(test::raises([] { Polygon(Tensor::from_data({0.0, 0.0, 0.0}, {1, 3})); },
                       Error::wrong_input_in_constructor),
          "linear_combination: an operand of three columns is refused");
    {
        ChecksScope off(false);
        check(test::unit_square().linear_combination(test::l_shape()).num_pieces() >= 2,
              "linear_combination: the same with the checks off");
    }
    return test::finish("polygon linear_combination");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
