// test_poly_zonotope_split_longest_gen - PolyZonotope split_longest_gen: the factor of the longest
//
// The longest dependent generator is the one of the largest sum of squares; the set is halved along
// the factor of its largest exponent, which split_dep_factor of that identifier gives as well.
//
// CORA: contSet/@polyZonotope/splitLongestGen.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
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
        // the second generator is the longer one: the factor 9 is halved
        const PolyZonotope P(column({0.0, 0.0}), Tensor::from_data({1, 0, 0, 2}, {2, 2}),
                             Tensor::zeros({2, 0}), {{1, 0}, {0, 1}}, {7, 9});
        const auto [L1, L2] = P.split_longest_gen();
        const auto [E1, E2] = P.split_dep_factor(9);
        check(close(L1.generators(), E1.generators()) && close(L2.center(), E2.center()),
              b + ": the factor of the longest generator");
    });
    check(test::raises(
              [] {
                  PolyZonotope(column({0.0}), Tensor::zeros({1, 0}), Tensor::zeros({1, 0}), {}, {})
                      .split_longest_gen();
              },
              Error::wrong_value),
          "split_longest_gen: no dependent generator");
    // with the checks off, a valid call still works
    ChecksScope off(false);
    const PolyZonotope P(column({0.0}), Tensor::from_data({1}, {1, 1}), Tensor::zeros({1, 0}),
                         {{1}}, {1});
    check(P.split_longest_gen().first.dim() == 1, "split_longest_gen: valid, checks off");
    return test::finish("poly_zonotope split_longest_gen");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
