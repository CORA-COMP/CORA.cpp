// test_poly_zonotope_compact - PolyZonotope compact: the canonical form
//
// CORA: contSet/@polyZonotope/compact_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {vec({1.0, 0.0}),
            Tensor::from_data({1, 1, 0, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {1, 2}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the factor 2 has no generator but the independent one: it goes
        const PolyZonotope P(vec({0.0}), Tensor::from_data({1.0, 0.0}, {2, 1}),
                             Tensor::zeros({0, 1}), {{1, 0}, {0, 0}}, {1, 2});
        const PolyZonotope C = P.compact();
        check(C.num_factors() == 1 && C.num_dependent() == 1,
              b + ": unused identifiers and zero generators go");
        // a constant generator moves to the center
        const PolyZonotope K(vec({1.0}), Tensor::from_data({2.0}, {1, 1}), Tensor::zeros({0, 1}),
                             {{0}}, {1});
        check(close(K.compact().center(), {3.0}, 0) && K.compact().num_dependent() == 0,
              b + ": a constant joins c");
    });

    // a polynomial zonotope has the method "all" only
    check(test::raises([&] { known().compact("fast"); }, Error::no_specific_alg),
          "compact: another method");
    ChecksScope off(false);
    check(known().compact().dim() == 2, "compact: valid with the checks off");
    return test::finish("poly_zonotope compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
