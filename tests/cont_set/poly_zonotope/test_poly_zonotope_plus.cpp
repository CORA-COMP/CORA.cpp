// test_poly_zonotope_plus - PolyZonotope plus: the sum with a zonotope, a point or a set
//
// CORA: contSet/@polyZonotope/plus.m
// PROPERTY: contSet/@polyZonotope/plus.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
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

/// Stated property: the sums of points of each summand lie in the zonotope of the sum.
void plus_property(const std::string &b) {
    Rng rng(3);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope A = test::random_poly(rng, 3), B = test::random_poly(rng, 3);
        const Tensor sum = test::sample(A, rng, 20) + test::sample(B, rng, 20);
        check(test::inside(A.plus(B).zonotope(), sum), b + ": the sums are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const Zonotope Z(column({1.0, 1.0}), Tensor::from_data({1, 0}, {2, 1}));
        const PolyZonotope S = P.plus(Z);
        check(close(S.center(), {2.0, 1.0}, 0) && S.independent_generators().shape()[1] == 2 &&
                  S.num_dependent() == P.num_dependent(),
              b + ": a zonotope joins the independent part");
        check(close(P.plus(column({1.0, 1.0})).center(), {2.0, 1.0}, 0), b + ": a point");
        // a polynomial zonotope brings factors of its own
        const PolyZonotope PP = P.plus(P);
        check(PP.num_factors() == 4 && PP.num_dependent() == 4,
              b + ": the factors of the second are new");
        check(plus(P, Z).is<PolyZonotope>(), b + ": through the registry");
        plus_property(b);
    });
    check(test::raises([] { known().plus(column({1.0})); }, Error::dimension_mismatch),
          "plus: point");
    check(test::raises(
              [] { known().plus(PolyZonotope(Zonotope(column({1.0}), Tensor::zeros({1, 0})))); },
              Error::dimension_mismatch),
          "plus: set");
    ChecksScope off(false);
    check(known().plus(column({1.0, 1.0})).dim() == 2, "plus: valid with the checks off");
    return test::finish("poly_zonotope plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
