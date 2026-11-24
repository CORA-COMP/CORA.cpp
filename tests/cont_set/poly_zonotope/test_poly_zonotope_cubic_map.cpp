// test_poly_zonotope_cubic_map - PolyZonotope cubic_map: the image under a cubic map
//
// The independent generators become dependent on new factors (absorbRounding), so the image is
// exact: its point at the factors (alpha, the old independent coefficients on the new factors) is
// the cubic map of the point of the set. One generator per ordered triple of columns. Lean-ORACLE
// debt: the request is in docs/oracle-requests.md.
//
// CORA: contSet/@polyZonotope/cubMap.m
// PROPERTY: contSet/@polyZonotope/cubMap.m
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

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// z_i = sum_k (k in ind[i]) x_k (x' T[i][k] x) at the point x of two coordinates.
std::vector<double> cubic(const std::vector<std::vector<Tensor>> &T, const std::vector<double> &x,
                          const std::vector<std::vector<int64_t>> &ind) {
    std::vector<double> z;
    for (std::size_t i = 0; i < T.size(); ++i) {
        double sum = 0.0;
        for (const int64_t k : ind[i]) {
            const std::vector<double> M = T[i][std::size_t(k)].data();
            double quad = 0.0;
            for (std::size_t r = 0; r < 2; ++r)
                for (std::size_t c = 0; c < 2; ++c) quad += x[r] * M[2 * r + c] * x[c];
            sum += x[std::size_t(k)] * quad;
        }
        z.push_back(sum);
    }
    return z;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(36);
        const PolyZonotope P = test::random_poly(rng, 2);
        std::vector<std::vector<Tensor>> T;
        for (int i = 0; i < 2; ++i) {
            T.emplace_back();
            for (int k = 0; k < 2; ++k) T.back().push_back(test::normal(rng, 2, 2));
        }
        // the exact image
        const PolyZonotope S = P.cubic_map(T);
        // [c, G] has 1 + 4 columns, the independent part and the error box add 2 + 2 more
        const int64_t N = 1 + 4 + 2 + 2;
        check(S.dim() == 2 && S.num_dependent() == N * N * N, b + ": a generator per triple");
        check(S.independent_generators().shape()[0] == 0 && S.num_factors() == 2 + 4,
              b + ": no independent generators, a factor per absorbed one");
        check(close(S.center(), S.center().zeros_like(), 0.0), b + ": the center is 0");
        // the point of the set, its factors at the new ids
        const std::vector<double> beta = {0.5, -0.25};
        const std::vector<double> x = test::point(P, {{1, 0.4}, {2, -0.8}}, beta);
        const std::map<int64_t, double> at = {{1, 0.4}, {2, -0.8}, {3, beta[0]}, {4, beta[1]}};
        const std::vector<std::vector<int64_t>> all = {{0, 1}, {0, 1}};
        check(close(Tensor::from_data(test::point(S, at, {}), {2}),
                    Tensor::from_data(cubic(T, x, all), {2}), 1e-12),
              b + ": the cubic map of the point");
        // ind: the matrices that do not enter are taken as zero
        const std::vector<std::vector<int64_t>> ind = {{1}, {0}};
        // the same point, the columns of ind only
        const PolyZonotope R = P.cubic_map(T, ind);
        check(close(Tensor::from_data(test::point(R, at, {}), {2}),
                    Tensor::from_data(cubic(T, x, ind), {2}), 1e-12),
              b + ": the columns of ind");
        // the refused arguments
        check(test::raises([&] { P.cubic_map({}); }, Error::wrong_value), b + ": no output");
        check(test::raises([&] { P.cubic_map({{T[0][0]}}); }, Error::dimension_mismatch),
              b + ": one matrix per coordinate");
        check(
            test::raises([&] { P.cubic_map({{test::normal(rng, 3, 3), test::normal(rng, 3, 3)}}); },
                         Error::dimension_mismatch),
            b + ": the shape of the matrices");
        check(test::raises([&] { P.cubic_map(T, {{0}}); }, Error::dimension_mismatch),
              b + ": ind per output");
        check(test::raises([&] { P.cubic_map(T, {{0, 2}, {0}}); }, Error::wrong_value),
              b + ": ind in range");
    });
    Rng rng(37);
    const PolyZonotope P = test::sample_poly();
    const std::vector<std::vector<Tensor>> T = {{test::normal(rng, 2, 2), test::normal(rng, 2, 2)}};
    ChecksScope off(false);
    check(P.cubic_map(T).dim() == 1, "cubic_map: valid, checks off");
    return test::finish("poly_zonotope cubic_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
