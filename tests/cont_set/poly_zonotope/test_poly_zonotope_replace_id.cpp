// test_poly_zonotope_replace_id - PolyZonotope replace_id: factors renamed
//
// Renaming 2 to 1 makes the two factors one: the points of the result at a are those of the set at
// (a, a). A rename to new identifiers keeps the points, the arguments are checked.
//
// CORA: contSet/@polyZonotope/replaceId.m
// PROPERTY: contSet/@polyZonotope/replaceId.m
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
        Rng rng(11);
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.replace_id({2}, {1});
        check(S.ids() == std::vector<int64_t>({1}), b + ": one factor is left");
        check(close(Tensor::from_data(test::point(S, {{1, 0.7}}, {0.2, 0.9}), {2}),
                    Tensor::from_data(test::point(P, {{1, 0.7}, {2, 0.7}}, {0.2, 0.9}), {2}),
                    1e-12),
              b + ": the two factors are one");
        const PolyZonotope R = P.replace_id({10, 20});
        check(R.ids() == std::vector<int64_t>({10, 20}), b + ": all identifiers renamed");
        check(close(Tensor::from_data(test::point(R, {{10, 0.3}, {20, -0.5}}, {0.1, 0.4}), {2}),
                    Tensor::from_data(test::point(P, {{1, 0.3}, {2, -0.5}}, {0.1, 0.4}), {2}),
                    1e-12),
              b + ": the points stay");
        // exponents add for the merged factors: (1,0) and (0,1) become (1) and (1), (2,1) -> (3)
        check(S.exponents()[2] == std::vector<int>({3}), b + ": exponents add");
    });
    check(test::raises([] { test::sample_poly().replace_id({1}, {3, 4}); }, Error::wrong_value),
          "replace_id: the lengths differ");
    check(test::raises([] { test::sample_poly().replace_id({8}, {3}); }, Error::wrong_value),
          "replace_id: an unknown identifier");
    ChecksScope off(false);
    check(test::sample_poly().replace_id({2}, {5}).ids()[1] == 5,
          "replace_id: valid with the checks off");
    return test::finish("poly_zonotope replace_id");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
