// test_poly_zonotope_resolve - PolyZonotope resolve: factors set to values
//
// The result at the remaining factors is the set with the resolved factors at the given values;
// a hand case with constants moving into the center. The default ids are all factors in order.
//
// Oracle debt: the request is in docs/oracle-requests.md (CORALean has the float definition).
//
// CORA: contSet/@polyZonotope/resolve.m
// PROPERTY: contSet/@polyZonotope/resolve.m
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
        Rng rng(25);
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.resolve(vec({0.5}), std::vector<int64_t>{2});
        check(S.ids() == std::vector<int64_t>({1}), b + ": the factor 2 is resolved");
        check(close(Tensor::from_data(test::point(S, {{1, -0.7}}, {0.3, 0.2}), {2}),
                    Tensor::from_data(test::point(P, {{1, -0.7}, {2, 0.5}}, {0.3, 0.2}), {2}),
                    1e-12),
              b + ": the points at the other factor");
        // c = 1, 2 a + 3 a b: with a = 2 and b = 3 the point 1 + 4 + 18 = 23, no factor left
        const PolyZonotope Q(vec({1.0}), Tensor::from_data({2, 3}, {2, 1}), Tensor::zeros({0, 1}),
                             {{1, 0}, {1, 1}}, {1, 2});
        // all factors given: a point
        const PolyZonotope V = Q.resolve(vec({2.0, 3.0}));
        check(V.num_dependent() == 0 && V.num_factors() == 0 && close(V.center(), {23.0}),
              b + ": all factors resolved");
        const PolyZonotope T = Q.resolve(vec({2.0}), std::vector<int64_t>{1});
        check(T.num_dependent() == 1 && close(T.center(), {5.0}) &&
                  close(T.generators(), Tensor::from_data({6.0}, {1, 1})),
              b + ": the constant joins the center");
    });
    const PolyZonotope P = test::sample_poly();
    check(test::raises([&] { P.resolve(vec({0.5, 0.5, 0.5})); }, Error::wrong_value),
          "resolve: the length");
    check(test::raises([&] { P.resolve(vec({0.5}), std::vector<int64_t>{9}); }, Error::wrong_value),
          "resolve: an unknown identifier");
    check(test::raises([&] { P.resolve(vec({0.5, 0.5}), std::vector<int64_t>{1, 1}); },
                       Error::wrong_value),
          "resolve: duplicates");
    ChecksScope off(false);
    check(P.resolve(vec({0.5, 0.5})).dim() == 2, "resolve: valid with the checks off");
    return test::finish("poly_zonotope resolve");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
