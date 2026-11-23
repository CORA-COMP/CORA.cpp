// test_poly_zonotope_restore_id - PolyZonotope restore_id: identifiers added to a set
//
// The missing identifiers are appended in the order given with zero exponents; the set and its
// points do not change.
//
// CORA: contSet/@polyZonotope/restoreId.m
// PROPERTY: contSet/@polyZonotope/restoreId.m
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
        Rng rng(7);
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.restore_id({2, 9, 4});
        check(S.ids() == std::vector<int64_t>({1, 2, 9, 4}), b + ": identifiers appended in order");
        // every generator has zero exponents for the new identifiers
        check(S.exponents().size() == P.exponents().size() && S.exponents()[2].size() == 4 &&
                  S.exponents()[2][2] == 0 && S.exponents()[2][3] == 0,
              b + ": zero exponents for the new identifiers");
        // the points of the set at the old factors stay
        check(close(Tensor::from_data(test::point(S, {{1, 0.2}, {2, -0.4}, {9, 1.0}}, {0.5, 0.5}),
                                      {2}),
                    Tensor::from_data(test::point(P, {{1, 0.2}, {2, -0.4}}, {0.5, 0.5}), {2}),
                    1e-12),
              b + ": the points stay");
        check(P.restore_id({}).ids() == P.ids(), b + ": nothing to restore");
    });
    check(test::raises([] { test::sample_poly().restore_id({7, 7}); }, Error::wrong_value),
          "restore_id: repeated identifiers");
    ChecksScope off(false);
    check(test::sample_poly().restore_id({5}).num_factors() == 3,
          "restore_id: valid with the checks off");
    return test::finish("poly_zonotope restore_id");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
