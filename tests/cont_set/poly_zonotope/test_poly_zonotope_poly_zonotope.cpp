// test_poly_zonotope_poly_zonotope - PolyZonotope constructor: merging and refusals
//
// CORA: contSet/@polyZonotope/polyZonotope.m
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

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

/// Equal exponents add, zero generators and zero independent generators go.
void merge_known(const std::string &b) {
    const PolyZonotope P(column({0.0, 0.0}), Tensor::from_data({1, 2, 0, 3, 5, 0}, {2, 3}),
                         Tensor::from_data({0, 0, 0, 0}, {2, 2}), {{1}, {1}, {2}}, {1});
    check(P.num_dependent() == 1 && P.independent_generators().shape()[1] == 0,
          b + ": the first two generators merge, the zero one and the zero independent ones go");
    check(close(P.generators(), {3.0, 8.0}, 0),
          b + ": the merged generator is the sum (1, 3) + (2, 5)");
    const PolyZonotope Z(Zonotope(column({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2})));
    check(Z.num_dependent() == 2 && Z.num_factors() == 2, b + ": a zonotope has E = I");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { merge_known(b); });
    check(test::raises(
              [] {
                  PolyZonotope(column({0.0}), Tensor::zeros({2, 1}), Tensor::zeros({1, 0}), {{1}},
                               {1});
              },
              Error::wrong_input_in_constructor),
          "constructor: generators of another dimension");
    // an exponent per identifier, and nonnegative exponents
    check(test::raises(
              [] {
                  PolyZonotope(column({0.0}), Tensor::zeros({1, 1}), Tensor::zeros({1, 0}),
                               {{1, 1}}, {1});
              },
              Error::wrong_input_in_constructor),
          "constructor: an exponent per identifier");
    // negative exponents are refused
    check(test::raises(
              [] {
                  PolyZonotope(column({0.0}), Tensor::zeros({1, 1}), Tensor::zeros({1, 0}), {{-1}},
                               {1});
              },
              Error::wrong_input_in_constructor),
          "constructor: negative exponents");
    ChecksScope off(false);
    check(known().dim() == 2 &&
              PolyZonotope(column({0.0}), Tensor::zeros({1, 0}), Tensor::zeros({1, 0}), {}, {})
                      .dim() == 1,
          "constructor: valid with the checks off");
    return test::finish("poly_zonotope poly_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
