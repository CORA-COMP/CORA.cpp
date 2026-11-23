// test_poly_zonotope_isequal - PolyZonotope isequal: equality up to a tolerance
//
// Equal data denote one set: a set equals itself and its compact form, a different generator or
// exponent breaks it, the generator order and the sign of independent generators do not matter,
// and renamed (reordered) identifiers are aligned. A zonotope or a box is compared as a polynomial
// zonotope.
//
// CORA: contSet/@polyZonotope/isequal.m
// PROPERTY: contSet/@polyZonotope/isequal.m
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
        Rng rng(15);
        const PolyZonotope P = test::random_poly(rng, 2);
        check(P.isequal(P), b + ": a set equals itself");
        check(P.isequal(P.compact()), b + ": and its compact form");
        // reordered generators and flipped independent generators
        const std::vector<int64_t> order = {3, 2, 1, 0};
        Exponents E;
        for (const int64_t j : order) E.push_back(P.exponents()[std::size_t(j)]);
        const PolyZonotope Q(P.center(), P.generators().take(order, -2),
                             P.independent_generators() * -1.0, E, P.ids());
        check(P.isequal(Q), b + ": order and the sign of independent generators");
        // swapped identifiers: the factor of identifier 1 is the second one
        Exponents Es;
        for (const std::vector<int> &col : P.exponents()) Es.push_back({col[1], col[0]});
        // the same set under the swapped identifiers, and another one
        check(P.isequal(
                  PolyZonotope(P.center(), P.generators(), P.independent_generators(), Es, {2, 1})),
              b + ": identifiers aligned");
        check(!P.isequal(
                  PolyZonotope(P.center(), P.generators(), P.independent_generators(), Es, {1, 2})),
              b + ": other exponents differ");
        // the center and the tolerance
        check(!P.isequal(PolyZonotope(P.center() + 1.0, P.generators(), P.independent_generators(),
                                      P.exponents(), P.ids())),
              b + ": other center");
        check(P.isequal(PolyZonotope(P.center() + 1e-9, P.generators(), P.independent_generators(),
                                     P.exponents(), P.ids()),
                        1e-6),
              b + ": within a tolerance");
        check(!P.isequal(test::random_poly(rng, 3)), b + ": other dimension");
        const Zonotope Z(vec({1.0, 2.0}), Tensor::from_data({1, 0, 0, 2}, {2, 2}));
        check(PolyZonotope(Z).isequal(Z) && !P.isequal(Z), b + ": a zonotope");
        const Interval I(vec({0.0, 1.0}), vec({2.0, 5.0}));
        check(PolyZonotope(I).isequal(I) && !P.isequal(I), b + ": a box");
    });
    check(test::raises([] { PolyZonotope::origin(2).isequal(PolyZonotope::origin(2), -1.0); },
                       Error::wrong_value),
          "isequal: tol < 0");
    ChecksScope off(false);
    check(PolyZonotope::origin(2).isequal(PolyZonotope::origin(2)), "isequal: valid, checks off");
    return test::finish("poly_zonotope isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
