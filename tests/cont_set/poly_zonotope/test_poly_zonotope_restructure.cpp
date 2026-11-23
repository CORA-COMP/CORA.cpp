// test_poly_zonotope_restructure - PolyZonotope restructure: the independent part boxed
//
// The independent generators become one diagonal generator per coordinate (the row sums of their
// magnitudes) on factors of their own, then n zero columns on factors of their own, as CORALean's
// restructure; the set is enclosed and nothing is independent.
// Bit for bit against CORALean: tests/global/oracle/test_lean_poly_zonotope_maps.cpp.
//
// CORA: contSet/@polyZonotope/restructure.m
// PROPERTY: contSet/@polyZonotope/restructure.m
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
        const PolyZonotope P = test::sample_poly();
        const PolyZonotope S = P.restructure();
        const int64_t n = P.dim(), h = P.num_dependent(), q = P.independent_generators().shape()[0];
        check(S.independent_generators().shape()[0] == 0, b + ": no independent generators");
        check(S.num_dependent() == h + 2 * n, b + ": the boxes and the zero columns are dependent");
        check(S.ids().size() == P.ids().size() + std::size_t(2 * n),
              b + ": a factor per new column");
        check(close(S.center(), P.center(), 0.0), b + ": the center stays");
        // the radius per coordinate is the row sum of the magnitudes
        const Tensor radius = P.independent_generators().abs().sum(-2);
        check(close(S.generators().slice(-2, h, h + n).sum(-2), radius, 1e-14),
              b + ": the box of the independent part");
        // every point of the set is in the new set (through the zonotope)
        Rng rng(2);
        check(test::inside(S.zonotope(), test::sample(P, rng, 20)), b + ": the set is enclosed");
        (void)q;
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.restructure().dim() == 2, "restructure: valid, checks off");
    return test::finish("poly_zonotope restructure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
