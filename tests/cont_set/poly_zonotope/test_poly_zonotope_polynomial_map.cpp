// test_poly_zonotope_polynomial_map - PolyZonotope polynomial_map: the image under a polynomial
//
// The independent generators become dependent on new factors (absorbRounding), so the image is
// exact: its point at the factors (alpha, the old independent coefficients on the new factors) is
// the polynomial at the point of the set. The terms are products of CORALean's mul, added by exact
// plus. Lean-ORACLE debt: evalPoly models the univariate case only, see docs/cora-differences.md.
//
// CORA: contSet/@polyZonotope/polyMap.m
// PROPERTY: contSet/@polyZonotope/polyMap.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
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
        Rng rng(35);
        const PolyZonotope P = test::random_poly(rng, 2);
        // z_0 = 2 x0 x1 - x0^2 + 1.5 x1^3 + 3, z_1 = -x0 x1 + 0.5 x0^2 + 2 x1^3 - 1
        const Tensor coeff = Tensor::from_data({2, -1, 1.5, 3, -1, 0.5, 2, -1}, {2, 4});
        const Exponents E = {{1, 1}, {2, 0}, {0, 3}, {0, 0}};
        const PolyZonotope S = P.polynomial_map(coeff, E);
        check(S.dim() == 2 && S.independent_generators().shape()[0] == 0,
              b + ": no independent generators");
        // the point of the set and of the image
        const std::vector<double> beta = {0.5, -0.25};
        const std::vector<double> x = test::point(P, {{1, 0.4}, {2, -0.8}}, beta);
        // the independent factors of P are the new factors 3 and 4 of the absorbed set
        const std::vector<double> z =
            test::point(S, {{1, 0.4}, {2, -0.8}, {3, beta[0]}, {4, beta[1]}}, {});
        // the polynomial at the point
        const double x0 = x[0], x1 = x[1];
        const std::vector<double> expected = {2 * x0 * x1 - x0 * x0 + 1.5 * x1 * x1 * x1 + 3,
                                              -x0 * x1 + 0.5 * x0 * x0 + 2 * x1 * x1 * x1 - 1};
        check(close(Tensor::from_data(z, {2}), Tensor::from_data(expected, {2}), 1e-12),
              b + ": the polynomial at the point of the set");
        // only a constant: the point
        const PolyZonotope C = P.polynomial_map(coeff.slice(-1, 3, 4), {{0, 0}});
        check(C.num_dependent() == 0 && close(C.center(), std::vector<double>{3, -1}, 0.0),
              b + ": a constant");
        // the identity: x0 and x1 as terms
        const PolyZonotope I =
            P.polynomial_map(Tensor::from_data({1, 0, 0, 1}, {2, 2}), {{1, 0}, {0, 1}});
        check(close(Tensor::from_data(
                        test::point(I, {{1, 0.4}, {2, -0.8}, {3, beta[0]}, {4, beta[1]}}, {}), {2}),
                    Tensor::from_data(x, {2}), 1e-12),
              b + ": the identity");
        // the refused arguments
        check(test::raises([&] { P.polynomial_map(coeff, {{1, 1}}); }, Error::wrong_value),
              b + ": one column per term");
        check(test::raises([&] { P.polynomial_map(coeff.slice(-1, 0, 1), {{1, 1, 1}}); },
                           Error::wrong_value),
              b + ": one exponent per coordinate");
        check(test::raises([&] { P.polynomial_map(coeff.slice(-1, 0, 1), {{-1, 1}}); },
                           Error::wrong_value),
              b + ": negative exponents");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.polynomial_map(Tensor::from_data({1.0, 1.0}, {1, 2}), {{1, 0}, {0, 1}}).dim() == 1,
          "polynomial_map: valid, checks off");
    return test::finish("poly_zonotope polynomial_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
