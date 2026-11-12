// test_poly_zonotope_compact_gens - PolyZonotope compact_gens: equal exponents added
//
// CORA: contSet/@polyZonotope/compact_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // columns 0, 2, 3 have the exponent (1, 0), 1 has (0, 1), 4 has (0, 0), 5 is zero (2, 1)
        const PolyZonotope P(column({1.0, 2.0}),
                             Tensor::from_data({1.0, 0.5, 0.25, 4.0, 1.0, 0.0, //
                                                -1.0, 3.0, 0.125, 8.0, 2.0, 0.0},
                                               {2, 6}),
                             Tensor::from_data({0.5, 0.0}, {2, 1}),
                             {{1, 0}, {0, 1}, {1, 0}, {1, 0}, {0, 0}, {2, 1}}, {1, 2},
                             detail::Verbatim{});
        const PolyZonotope C = P.compact_gens();
        check(C.num_dependent() == 4, b + ": one column per exponent column");
        const std::vector<double> g = C.generators().data();
        // the least index keeps the sum, the other columns of its class are added by the last first
        check(g[0] == 1.0 + (0.25 + 4.0) && g[1] == 0.5 && g[2] == 1.0 && g[3] == 0.0,
              b + ": the first row of the merged generators");
        check(g[4] == -1.0 + (0.125 + 8.0) && g[5] == 3.0 && g[6] == 2.0 && g[7] == 0.0,
              b + ": the second row of the merged generators");
        check(C.exponents()[2] == std::vector<int>({0, 0}) &&
                  C.exponents()[3] == std::vector<int>({2, 1}),
              b + ": the constant column and the zero column stay, in their order");
        check(C.num_factors() == 2 && close(C.center(), {1.0, 2.0}, 0) &&
                  C.independent_generators().shape()[1] == 1,
              b + ": the factors, the center and the independent generators stay");
        check(P.compact().num_dependent() < C.num_dependent(),
              b + ": compact drops what this keeps");
        // distinct exponents: the set is returned as it is
        const PolyZonotope D(column({0.0}), Tensor::from_data({1.0, 0.0}, {1, 2}),
                             Tensor::zeros({1, 0}), {{1}, {2}}, {1}, detail::Verbatim{});
        check(D.compact_gens().num_dependent() == 2,
              b + ": distinct exponents (and a zero column) stay");
    });
    ChecksScope off(false);
    return test::finish("poly_zonotope compact_gens");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
