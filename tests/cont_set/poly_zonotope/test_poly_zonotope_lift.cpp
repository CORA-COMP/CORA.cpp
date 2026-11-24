// test_poly_zonotope_lift - PolyZonotope lift: the coordinates permuted
//
// State i goes to position dims[i], as the zonotope's lift (CORA's lift_ projects onto dims, the
// inverse permutation); the factors and exponents stay. A lift to new dimensions is refused, as
// they would be unbounded.
// Bit for bit against CORALean: tests/global/oracle/test_lean_poly_zonotope_ids.cpp.
//
// CORA: contSet/@polyZonotope/lift_.m
// PROPERTY: contSet/@polyZonotope/lift_.m
// CORA authors: Mark Wetzlinger
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
        Rng rng(32);
        const PolyZonotope P = test::random_poly(rng, 3);
        const PolyZonotope L = P.lift(3, {2, 0, 1});
        // state 0 goes to position 2: the new coordinate j is the old one at source[j]
        check(close(L.center().slice(-1, 2, 3), P.center().slice(-1, 0, 1), 0) &&
                  close(L.center().slice(-1, 0, 1), P.center().slice(-1, 1, 2), 0) &&
                  close(L.center().slice(-1, 1, 2), P.center().slice(-1, 2, 3), 0),
              b + ": the center is permuted");
        check(L.exponents() == P.exponents() && L.ids() == P.ids(), b + ": the factors stay");
        const std::vector<double> px = test::point(P, {{1, 0.3}, {2, -0.6}}, {0.5, 0.25}),
                                  pl = test::point(L, {{1, 0.3}, {2, -0.6}}, {0.5, 0.25});
        check(pl[2] == px[0] && pl[0] == px[1] && pl[1] == px[2], b + ": the points are permuted");
        check(close(P.lift(3, {0, 1, 2}).generators(), P.generators(), 0), b + ": the identity");
        check(test::raises([&] { P.lift(2, {0, 1, 2}); }, Error::wrong_value),
              b + ": N below the dimension");
        check(test::raises([&] { P.lift(3, {0, 1, 1}); }, Error::wrong_value),
              b + ": a position twice");
        const PolyZonotope Q = test::random_poly(rng, 2);
        check(test::raises([&] { Q.lift(3, {0, 2}); }, Error::not_defined),
              b + ": new dimensions would be unbounded");
        // through the generic interface
        check(static_cast<const ContSet &>(P)
                      .lift(3, std::vector<int64_t>{1, 2, 0})
                      .as<PolyZonotope>()
                      .dim() == 3,
              b + ": lift_ of a ContSet");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.lift(2, {1, 0}).dim() == 2, "lift: valid, checks off");
    return test::finish("poly_zonotope lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
