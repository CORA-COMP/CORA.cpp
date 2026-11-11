// test_poly_zonotope_reduce - PolyZonotope reduce: fewer generators, a superset
//
// CORA: contSet/@polyZonotope/reduce.m
// PROPERTY: contSet/@polyZonotope/reduce.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
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

/// Stated property: the reduced set has at most order * n generators and holds the points.
void reduce_property(const std::string &b) {
    Rng rng(17);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope R = P.reduce(2);
        check(R.num_dependent() + R.independent_generators().shape()[1] <= 4,
              b + ": at most 2 n generators");
        check(test::inside(R.zonotope(), test::sample(P, rng, 40)), b + ": the points are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        check(P.reduce(5).num_dependent() == P.num_dependent(),
              b + ": below the order nothing changes");
        const PolyZonotope R = P.reduce(1);
        check(R.num_dependent() == 0 && R.independent_generators().shape()[1] == 2,
              b + ": order 1: a box");
        reduce_property(b);
    });
    check(test::raises([] { known().reduce(0); }, Error::wrong_value), "reduce: the order");
    ChecksScope off(false);
    check(known().reduce(2).dim() == 2, "reduce: valid with the checks off");
    return test::finish("poly_zonotope reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
