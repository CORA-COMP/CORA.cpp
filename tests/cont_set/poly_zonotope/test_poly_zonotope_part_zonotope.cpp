// test_poly_zonotope_part_zonotope - PolyZonotope part_zonotope: groups by exponent pattern
//
// The groups are the distinct patterns of the chosen factors in the order of their last
// occurrence; each group but the zero pattern gets a new factor standing for the monomial: its
// value (an odd pattern), or 2 m - 1 for an all-even pattern, whose generators are halved with a
// copy of the halves. At those values the point of the result is the point of the set.
// Lean-ORACLE debt: the request is in docs/oracle-requests.md.
//
// CORA: contSet/@polyZonotope/partZonotope.m
// PROPERTY: contSet/@polyZonotope/partZonotope.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
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
        Rng rng(38);
        // the patterns of factor 1: 1, 0, 2, 1; by last occurrence the groups are 0, 2, 1
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope S = P.part_zonotope({1});
        check(S.num_dependent() == 8 && S.ids() == std::vector<int64_t>({1, 2, 3, 4, 5}),
              b + ": two blocks, a new factor per pattern");
        check(S.exponents()[0] == std::vector<int>({0, 0, 0, 0, 1}) &&
                  S.exponents()[1] == std::vector<int>({0, 1, 0, 0, 0}) &&
                  S.exponents()[2] == std::vector<int>({0, 1, 0, 1, 0}) &&
                  S.exponents()[3] == std::vector<int>({0, 2, 0, 0, 1}),
              b + ": the chosen factor is gone, the new factor of the pattern is 1");
        check(close(S.generators().slice(-2, 2, 3), P.generators().slice(-2, 2, 3) * 0.5, 0.0) &&
                  close(S.generators().slice(-2, 6, 7), P.generators().slice(-2, 2, 3) * 0.5, 0.0),
              b + ": an even pattern is halved, with a copy");
        check(
            close(S.generators().slice(-2, 5, 6), S.generators().slice(-2, 5, 6).zeros_like(), 0.0),
            b + ": the other copies are zero");
        // the point: new factors at the monomials of factor 1 (value a)
        const double a = 0.7, a2 = -0.4;
        const std::vector<double> beta = {0.5, -0.25};
        const std::vector<double> x = test::point(P, {{1, a}, {2, a2}}, beta),
                                  z = test::point(S, {{2, a2}, {4, 2 * a * a - 1}, {5, a}}, beta);
        check(close(Tensor::from_data(z, {2}), Tensor::from_data(x, {2}), 1e-12),
              b + ": the point of the set");
        // all patterns even: x^2 and x^4 are two groups, both shifted
        const PolyZonotope Q(vec({0.5}), Tensor::from_data({1.0, -2.0}, {2, 1}),
                             Tensor::zeros({0, 1}), {{2}, {4}}, {1});
        const PolyZonotope R = Q.part_zonotope({1});
        const std::vector<double> qx = test::point(Q, {{1, a}}, {});
        const std::vector<double> rz =
            test::point(R, {{2, 2 * a * a - 1}, {3, 2 * std::pow(a, 4) - 1}}, {});
        check(close(Tensor::from_data(rz, {1}), Tensor::from_data(qx, {1}), 1e-12),
              b + ": all patterns even");
        check(test::raises([&] { P.part_zonotope({9}); }, Error::wrong_value),
              b + ": an identifier the set lacks");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.part_zonotope({1}).dim() == 2, "part_zonotope: valid, checks off");
    return test::finish("poly_zonotope part_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
