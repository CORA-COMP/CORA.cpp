// test_poly_zonotope_compact - PolyZonotope compact: the canonical form
//
// CORA: contSet/@polyZonotope/compact_.m
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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the factor 2 has no generator but the independent one: it goes
        const PolyZonotope P(column({0.0}), Tensor::from_data({1.0, 0.0}, {1, 2}),
                             Tensor::zeros({1, 0}), {{1, 0}, {0, 0}}, {1, 2});
        const PolyZonotope C = P.compact();
        check(C.num_factors() == 1 && C.num_dependent() == 1,
              b + ": unused identifiers and zero generators go");
        // a constant generator moves to the center
        const PolyZonotope K(column({1.0}), Tensor::from_data({2.0}, {1, 1}), Tensor::zeros({1, 0}),
                             {{0}}, {1});
        check(close(K.compact().center(), {3.0}, 0) && K.compact().num_dependent() == 0,
              b + ": a constant joins c");
    });

    ChecksScope off(false);
    check(known().compact().dim() == 2, "compact: valid with the checks off");
    return test::finish("poly_zonotope compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
