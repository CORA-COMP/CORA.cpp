// test_polygon_unite - Polygon::unite: the union: the pieces of both
//
// CORA: contSet/@polygon/or.m
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

void union_of_two(const std::string &b) {
    const Polygon U = test::unit_square().unite(test::l_shape());
    check(U.num_pieces() == 1 + test::l_shape().num_pieces() && close(U.volume(), 3.0),
          b + ": the square inside the L adds no area");
    const Set S = Set(test::unit_square()) | Set(test::boundary({{3, 3}, {4, 3}, {3, 4}}));
    check(close(S.as<Polygon>().volume(), 1.5), b + ": the operator |");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { union_of_two(b); });
    // an invalid operand is refused before the operation
    check(test::raises([] { Polygon(Tensor::from_data({0.0, 0.0, 0.0}, {1, 3})); },
                       Error::wrong_input_in_constructor),
          "unite: an operand of three columns is refused");
    {
        ChecksScope off(false);
        check(test::unit_square().unite(test::unit_square()).num_pieces() == 2,
              "unite: the same with the checks off");
    }
    return test::finish("polygon unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
