// test_poly_zonotope_matrix_map - PolyZonotope matrix_map: the product of two matrix-valued sets
//
// The sets hold an n x k and a k x m matrix column by column. At equal factors the product of the
// matrices is a point of the result when each independent generator of a product pair takes the
// product of the coefficients of its pair (the dependent generators are exact). Lean-ORACLE debt:
// the request is in docs/oracle-requests.md.
//
// CORA: contSet/@polyZonotope/matMap.m
// PROPERTY: contSet/@polyZonotope/matMap.m
// CORA authors: Tobias Ladner
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

/// The monomial of generator j of P at the factor values alpha (0 where not given); j = -1 is the
/// center column, whose monomial is 1.
double monomial(const PolyZonotope &P, int64_t j, const std::map<int64_t, double> &alpha) {
    if (j < 0) return 1.0;
    double m = 1.0;
    for (std::size_t r = 0; r < P.ids().size(); ++r) {
        const auto at = alpha.find(P.ids()[r]);
        m *= std::pow(at == alpha.end() ? 0.0 : at->second, P.exponents()[std::size_t(j)][r]);
    }
    return m;
}

/// The column-by-column vector of an n x k matrix product (n, k, m) of two such vectors.
std::vector<double> product(const std::vector<double> &a, const std::vector<double> &b, int n,
                            int k, int m) {
    std::vector<double> c(std::size_t(n * m), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int l = 0; l < k; ++l)
                c[std::size_t(i + n * j)] += a[std::size_t(i + n * l)] * b[std::size_t(l + k * j)];
    return c;
}

/// The coefficients of the independent generators of the product, from those of the factors.
std::vector<double> pair_betas(const PolyZonotope &P1, const PolyZonotope &P2,
                               const std::map<int64_t, double> &alpha,
                               const std::vector<double> &b1, const std::vector<double> &b2) {
    std::vector<double> ext1{1.0}, ext2{1.0}, out;
    for (int64_t j = 0; j < P1.num_dependent(); ++j) ext1.push_back(monomial(P1, j, alpha));
    for (int64_t j = 0; j < P2.num_dependent(); ++j) ext2.push_back(monomial(P2, j, alpha));
    for (const double x : b1)
        for (const double y : ext2) out.push_back(x * y);
    for (const double x : b1)
        for (const double y : b2) out.push_back(x * y);
    for (const double x : ext1)
        for (const double y : b2) out.push_back(x * y);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(33);
        const int n = 2, k = 2, m = 3;
        const PolyZonotope P1 = test::random_poly(rng, n * k), P2 = test::random_poly(rng, k * m);
        const PolyZonotope S = P1.matrix_map(P2, n, k, m);
        check(S.dim() == n * m, b + ": the dimension of the product");
        check(S.num_dependent() == (4 + 1) * (4 + 1), b + ": a generator per pair of columns");
        check(S.independent_generators().shape()[0] == 2 * 5 + 2 * 2 + 5 * 2,
              b + ": the three blocks of independent pairs");
        check(close(S.center(), S.center().zeros_like(), 0.0), b + ": the center is 0");
        const std::map<int64_t, double> alpha = {{1, 0.4}, {2, -0.8}};
        const std::vector<double> b1 = {0.5, -0.25}, b2 = {-0.75, 1.0};
        // the product of the points
        const std::vector<double> x1 = test::point(P1, alpha, b1), x2 = test::point(P2, alpha, b2);
        const std::vector<double> z = test::point(S, alpha, pair_betas(P1, P2, alpha, b1, b2)),
                                  expected = product(x1, x2, n, k, m);
        check(close(Tensor::from_data(z, {n * m}), Tensor::from_data(expected, {n * m}), 1e-12),
              b + ": the product of the points is a point of the product");
        // identifiers that differ: the sets are placed over the union
        const PolyZonotope Q2(P2.center(), P2.generators(), P2.independent_generators(),
                              P2.exponents(), {2, 3});
        // the product over three identifiers
        const PolyZonotope R = P1.matrix_map(Q2, n, k, m);
        check(R.ids() == std::vector<int64_t>({1, 2, 3}), b + ": the union of the identifiers");
        // the point at the union of the identifiers
        const std::map<int64_t, double> alpha3 = {{1, 0.4}, {2, -0.8}, {3, 0.6}};
        const std::map<int64_t, double> alpha1 = {{1, 0.4}, {2, -0.8}},
                                        alpha2 = {{2, 0.4}, {3, -0.8}};
        const std::vector<double> y1 = test::point(P1, alpha1, b1),
                                  y2 = test::point(Q2, {{2, -0.8}, {3, 0.6}}, b2);
        (void)alpha2;
        const std::vector<double> w = test::point(R, alpha3, pair_betas(P1, Q2, alpha3, b1, b2));
        check(close(Tensor::from_data(w, {n * m}),
                    Tensor::from_data(product(y1, y2, n, k, m), {n * m}), 1e-12),
              b + ": the same with different identifiers");
        // the refused dimensions
        check(test::raises([&] { P1.matrix_map(P2, n, k, m + 1); }, Error::wrong_value),
              b + ": the second dimension");
        check(test::raises([&] { P1.matrix_map(P2, n + 1, k, m); }, Error::wrong_value),
              b + ": the first dimension");
    });
    Rng rng(34);
    const PolyZonotope P1 = test::random_poly(rng, 4), P2 = test::random_poly(rng, 4);
    ChecksScope off(false);
    check(P1.matrix_map(P2, 2, 2, 2).dim() == 4, "matrix_map: valid, checks off");
    return test::finish("poly_zonotope matrix_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
