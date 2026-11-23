// test_poly_zonotope_rm_id - PolyZonotope rm_id: factors removed from a set
//
// The factors are fixed to 1: the points of the result at the remaining factors are those of the
// set with the removed ones at 1. Unknown identifiers are ignored.
//
// CORA: contSet/@polyZonotope/rmId.m
// PROPERTY: contSet/@polyZonotope/rmId.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(9);
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.rm_id({1});
        check(S.ids() == std::vector<int64_t>({2}), b + ": the identifier 1 is gone");
        check(close(Tensor::from_data(test::point(S, {{2, 0.6}}, {0.1, -0.3}), {2}),
                    Tensor::from_data(test::point(P, {{1, 1.0}, {2, 0.6}}, {0.1, -0.3}), {2}),
                    1e-12),
              b + ": the removed factor is 1");
        // the exponents become 0, 1, 1, 2 over the factor 2: the two of exponent 1 add
        check(S.num_dependent() == 3, b + ": generators of equal exponent are added");
        check(P.rm_id({}).num_dependent() == P.num_dependent(), b + ": nothing removed");
    });
    check(test::raises([] { test::sample_poly().rm_id({77}); }, Error::wrong_value),
          "rm_id: an unknown identifier");
    ChecksScope off(false);
    check(test::sample_poly().rm_id({2}).num_factors() == 1, "rm_id: valid with the checks off");
    return test::finish("poly_zonotope rm_id");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
