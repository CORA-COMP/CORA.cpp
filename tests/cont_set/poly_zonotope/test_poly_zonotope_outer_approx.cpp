// test_poly_zonotope_outer_approx - PolyZonotope outer_approx: small generators boxed
//
// The dependent generators below the threshold (sorted by size, the even ones halved for the
// sort) leave the set and their magnitudes are added to a diagonal box in the independent block;
// the result contains the set. Lean-ORACLE debt: the request is in docs/oracle-requests.md.
//
// CORA: contSet/@polyZonotope/outerApprox.m
// PROPERTY: contSet/@polyZonotope/outerApprox.m
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
        // generators (0.1, 0), (0, 0.01) and (1, 1) on the monomials a, a^2, a^3
        const Tensor G = Tensor::from_data({0.1, 0.0, 0.0, 0.01, 1.0, 1.0}, {3, 2});
        const PolyZonotope Q(vec({0.0, 0.0}), G, Tensor::zeros({0, 2}), {{1}, {2}, {3}}, {1});
        // the order is a^2 (halved: 0.005), a, then (1, 1): the first two fit within 0.2
        const PolyZonotope S = Q.outer_approx(0.2, {});
        check(S.num_dependent() == 1 && S.exponents()[0] == std::vector<int>({3}),
              b + ": the two small generators leave");
        check(close(S.independent_generators(), std::vector<double>{0.1, 0.0, 0.0, 0.01}, 1e-15),
              b + ": their magnitudes form the box");
        Rng rng(51);
        check(test::inside(S.zonotope(), test::sample(Q, rng, 30)), b + ": the set is enclosed");
        // tol 0: nothing is taken, the box is zero
        const PolyZonotope Z = Q.outer_approx(0.0, {});
        check(Z.num_dependent() == 3 &&
                  close(Z.independent_generators(), Tensor::zeros({2, 2}), 0.0),
              b + ": tol 0 takes nothing");
        // a large tol takes everything that depends on a factor outside ids
        check(Q.outer_approx(10.0, {}).num_dependent() == 0, b + ": a large tol takes all");
        check(Q.outer_approx(10.0, {1}).num_dependent() == 3, b + ": factors in ids are kept");
        // the independent generators are taken as well, and stay
        const PolyZonotope P = test::random_poly(rng, 2);
        const PolyZonotope R = P.outer_approx(10.0, {});
        check(R.independent_generators().shape()[0] == 2 + 2, b + ": the independent ones stay");
        const Tensor radius =
            P.generators().abs().sum(-2) + P.independent_generators().abs().sum(-2);
        check(close(R.independent_generators().slice(-2, 2, 4).sum(-2), radius, 1e-12),
              b + ": the box of all the generators");
        check(test::inside(R.zonotope(), test::sample(P, rng, 30)),
              b + ": the random set is enclosed");
        check(test::raises([&] { P.outer_approx(-1.0, {}); }, Error::wrong_value),
              b + ": a negative tol");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.outer_approx(0.5, {1}).dim() == 2, "outer_approx: valid, checks off");
    return test::finish("poly_zonotope outer_approx");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
