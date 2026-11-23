// test_poly_zonotope_linear_combination - PolyZonotope linear_combination: the segments
//
// The result lays the factors of the sets side by side with a new factor t last: its point at the
// factors (alpha, alpha', t) and the independent coefficients paired as CORA's zonotope enclose
// pairs them is (1 + t)/2 x + (1 - t)/2 y for the points x, y of the sets at alpha, beta and
// alpha', beta'.
// Bit for bit against CORALean: tests/global/oracle/test_lean_poly_zonotope_maps.cpp.
//
// CORA: contSet/@polyZonotope/linComb.m
// PROPERTY: contSet/@polyZonotope/linComb.m
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
        Rng rng(31);
        const PolyZonotope P = test::random_poly(rng, 2), Q = test::random_poly(rng, 2);
        const PolyZonotope S = P.linear_combination(Q);
        check(S.dim() == 2 && S.num_factors() == 5, b + ": the factors side by side, t last");
        check(S.ids() == std::vector<int64_t>({1, 2, 3, 4, 5}), b + ": the identifiers 1, 2, ...");
        check(S.num_dependent() == 2 * (4 + 4) + 1, b + ": two blocks and the center column");
        check(close(S.center(), (P.center() + Q.center()) * 0.5, 0.0), b + ": the midpoint");
        // the points: x at (0.4, -0.8, beta 0.5, -0.25), y at (0.3, 0.6, beta -0.75, 1), t = 0.4
        const double t = 0.4, b1[2] = {0.5, -0.25}, b2[2] = {-0.75, 1.0};
        const std::vector<double> x = test::point(P, {{1, 0.4}, {2, -0.8}}, {b1[0], b1[1]}),
                                  y = test::point(Q, {{1, 0.3}, {2, 0.6}}, {b2[0], b2[1]});
        std::vector<double> beta(4), expected(2);
        for (std::size_t j = 0; j < 2; ++j) {
            beta[j] = ((1 + t) * b1[j] + (1 - t) * b2[j]) / 2;
            beta[2 + j] = ((1 + t) * b1[j] - (1 - t) * b2[j]) / 2;
            expected[j] = (1 + t) / 2 * x[j] + (1 - t) / 2 * y[j];
        }
        const std::vector<double> z =
            test::point(S, {{1, 0.4}, {2, -0.8}, {3, 0.3}, {4, 0.6}, {5, t}}, beta);
        check(close(Tensor::from_data(z, {2}), Tensor::from_data(expected, {2}), 1e-12),
              b + ": the point of the combination");
        // t = 1 gives the first set, t = -1 the second, the independent parts as they are
        const std::vector<double> at_one = test::point(
            S, {{1, 0.4}, {2, -0.8}, {3, 0.3}, {4, 0.6}, {5, 1.0}}, {b1[0], b1[1], b1[0], b1[1]});
        check(close(Tensor::from_data(at_one, {2}), Tensor::from_data(x, {2}), 1e-12),
              b + ": t = 1 is the first set");
        // unequal independent blocks: the zonotope has none; both sets are in the zonotope hull
        const Zonotope Z(Q.center(), Q.generators());
        const PolyZonotope T = P.linear_combination(Z);
        const Zonotope hull = T.zonotope();
        check(T.independent_generators().shape()[0] == 2, b + ": the surplus block is copied");
        check(test::inside(hull, test::sample(P, rng, 30)), b + ": the first set is enclosed");
        check(test::inside(hull, test::sample(PolyZonotope(Z), rng, 30)),
              b + ": the second set is enclosed");
        const PolyZonotope U = PolyZonotope(Z).linear_combination(P);
        check(test::inside(U.zonotope(), test::sample(P, rng, 30)), b + ": the longer one second");
        const Interval box(Tensor::from_data({-1.0, 0.0}, {2}), Tensor::from_data({1.0, 2.0}, {2}));
        check(P.linear_combination(box).dim() == 2, b + ": a box");
        check(test::raises([&] { P.linear_combination(test::random_poly(rng, 3)); },
                           Error::dimension_mismatch),
              b + ": different dimensions");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.linear_combination(P).dim() == 2, "linear_combination: valid, checks off");
    return test::finish("poly_zonotope linear_combination");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
