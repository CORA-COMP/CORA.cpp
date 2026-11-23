// test_poly_zonotope_no_indep - PolyZonotope no_indep: the set without independent generators
//
// The dependent part is kept, the independent generators go: the points of the result are the
// points of the set at beta = 0.
//
// CORA: contSet/@polyZonotope/noIndep.m
// PROPERTY: contSet/@polyZonotope/noIndep.m
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
        Rng rng(3);
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.no_indep();
        check(S.independent_generators().shape()[0] == 0 && S.num_dependent() == P.num_dependent(),
              b + ": no independent generators, the dependent ones stay");
        check(close(S.generators(), P.generators(), 0) && S.ids() == P.ids(),
              b + ": generators and identifiers unchanged");
        check(close(Tensor::from_data(test::point(S, {{1, 0.3}, {2, -0.7}}, {}), {2}),
                    Tensor::from_data(test::point(P, {{1, 0.3}, {2, -0.7}}, {0.0, 0.0}), {2}),
                    1e-12),
              b + ": the points are those at beta = 0");
    });
    ChecksScope off(false);
    check(test::sample_poly().no_indep().dim() == 2, "no_indep: valid with the checks off");
    return test::finish("poly_zonotope no_indep");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
