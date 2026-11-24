// test_poly_zonotope_convex_hull - PolyZonotope convex_hull: the chords or the hull of two sets
//
// With one set the result is its linear combination with itself; with two it contains both
// (their samples lie in its zonotope enclosure) and has the independent block of the pairing with
// the zero column.
// Bit for bit against CORALean: tests/global/oracle/test_lean_poly_zonotope_maps.cpp.
//
// CORA: contSet/@polyZonotope/convHull_.m
// PROPERTY: contSet/@polyZonotope/convHull_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(41);
        const PolyZonotope P = test::random_poly(rng, 2), Q = test::random_poly(rng, 2);
        // one set: the chords with itself
        const PolyZonotope chords = P.convex_hull(), same = P.linear_combination(P);
        check(chords.exponents() == same.exponents() &&
                  close(chords.generators(), same.generators(), 0.0),
              b + ": one argument is the linear combination with itself");
        // two sets: the chords of each, then with each other
        const PolyZonotope H = P.convex_hull(Q);
        check(H.independent_generators().shape()[0] == 2 + 2 + 1,
              b + ": the pairing and a zero column");
        check(close(H.independent_generators().slice(-2, 4, 5), Tensor::zeros({1, 2}), 0.0),
              b + ": the last column is zero");
        check(H.num_factors() == 11, b + ": the factors of the chord sets");
        // both sets are in the hull
        const Zonotope hull = H.zonotope();
        check(test::inside(hull, test::sample(P, rng, 30)), b + ": the first set is enclosed");
        check(test::inside(hull, test::sample(Q, rng, 30)), b + ": the second set is enclosed");
        // a zonotope and a box
        const Zonotope Z(Q.center(), Q.generators());
        check(P.convex_hull(Z).dim() == 2 && P.convex_hull(Q.interval()).dim() == 2,
              b + ": a zonotope and a box");
        check(test::raises([&] { P.convex_hull(test::random_poly(rng, 3)); },
                           Error::dimension_mismatch),
              b + ": different dimensions");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.convex_hull(P).dim() == 2, "convex_hull: valid, checks off");
    return test::finish("poly_zonotope convex_hull");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
