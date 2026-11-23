// test_poly_zonotope_only_id - PolyZonotope only_id: the part that depends on given factors
//
// The two parts add to the set: the point of the first at the factors ids plus that of the second
// is the point of the set. The center is in the first part, the independent generators in the
// second.
//
// CORA: contSet/@polyZonotope/onlyId.m
// PROPERTY: contSet/@polyZonotope/onlyId.m
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
        // c = (1, 2), a (1, 0), b (0, 1), a b (1, 1), t (2, 2)
        const PolyZonotope P(vec({1.0, 2.0}), Tensor::from_data({1, 0, 0, 1, 1, 1}, {3, 2}),
                             Tensor::from_data({2, 2}, {1, 2}), {{1, 0}, {0, 1}, {1, 1}}, {4, 5});
        const auto [A, R] = P.only_id({4});
        check(A.ids() == std::vector<int64_t>({4}) && A.num_dependent() == 1 &&
                  close(A.center(), {1.0, 2.0}),
              b + ": the first part is a, with the center");
        check(R.num_dependent() == 2 && R.independent_generators().shape()[0] == 1 &&
                  close(R.center(), {0.0, 0.0}),
              b + ": the rest: b, ab and the independent generator");
        // the points of the parts at the same factors add up to the point of the set
        const std::map<int64_t, double> alpha = {{4, 0.5}, {5, -0.25}};
        const std::vector<double> pa = test::point(A, {{4, 0.5}}, {}),
                                  pr = test::point(R, alpha, {0.75});
        std::vector<double> sum = {pa[0] + pr[0], pa[1] + pr[1]};
        check(close(Tensor::from_data(sum, {2}),
                    Tensor::from_data(test::point(P, alpha, {0.75}), {2}), 1e-12),
              b + ": the parts add to the set");
        // the same on random sets
        Rng rng(4);
        for (int k = 0; k < 5; ++k) {
            const PolyZonotope Q = test::random_poly(rng, 2);
            const auto [A1, R1] = Q.only_id({2});
            const std::map<int64_t, double> a = {{1, 0.35}, {2, -0.45}};
            const std::vector<double> x = test::point(A1, a, {}),
                                      y = test::point(R1, a, {0.2, -0.6}),
                                      z = test::point(Q, a, {0.2, -0.6});
            check(close(Tensor::from_data({x[0] + y[0], x[1] + y[1]}, {2}),
                        Tensor::from_data(z, {2}), 1e-12),
                  b + ": random sets, the parts add");
        }
        // all identifiers: the whole dependent part in the first set
        const auto [All, None] = P.only_id({4, 5});
        check(All.num_dependent() == 3 && None.num_dependent() == 0, b + ": all identifiers");
    });
    const PolyZonotope Q = test::sample_poly();
    check(test::raises([&] { Q.only_id({1, 1}); }, Error::wrong_value), "only_id: duplicates");
    check(test::raises([&] { Q.only_id({3}); }, Error::wrong_value), "only_id: unknown identifier");
    ChecksScope off(false);
    check(Q.only_id({1}).first.dim() == 2, "only_id: valid with the checks off");
    return test::finish("poly_zonotope only_id");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
