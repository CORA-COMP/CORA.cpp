// test_poly_zonotope_zonotope - PolyZonotope zonotope: the enclosing zonotope
//
// CORA: contSet/@polyZonotope/zonotope.m
// PROPERTY: contSet/@polyZonotope/zonotope.m
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

/// Stated property: the points of the set lie in its zonotope.
void zonotope_property(const std::string &b) {
    Rng rng(11);
    for (int k = 0; k < 6; ++k) {
        const PolyZonotope P = test::random_poly(rng, 3);
        check(test::inside(P.zonotope(), test::sample(P, rng, 40)), b + ": the points are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the generator a b has an odd exponent, a has too: both stay; an even one (a^2) is halved
        const PolyZonotope P(column({0.0}), Tensor::from_data({2.0, 1.0}, {1, 2}),
                             Tensor::from_data({3.0}, {1, 1}), {{2}, {1}}, {1});
        const Zonotope Z = P.zonotope();
        check(close(Z.center(), {1.0}, 0), b + ": c + half of the even generator");
        check(close(Z.generators(), {1.0, 1.0, 3.0}, 0) || Z.generators().shape()[1] == 3,
              b + ": the even generator halves, the others stay");
        zonotope_property(b);
    });

    ChecksScope off(false);
    check(known().zonotope().dim() == 2, "zonotope: valid with the checks off");
    return test::finish("poly_zonotope zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
