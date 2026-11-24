// test_poly_zonotope_jacobian - PolyZonotope jacobian: the derivative along each factor
//
// The point of the derivative along the factor k at the factors alpha is the central difference
// of the dependent part along k; the center is 0, the independent part is gone, the exponent of k
// is lowered by one (a constant stays a zero column).
// Bit for bit against CORALean: tests/global/oracle/test_lean_poly_zonotope_maps.cpp.
//
// CORA: contSet/@polyZonotope/jacobian.m
// PROPERTY: contSet/@polyZonotope/jacobian.m
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
        Rng rng(42);
        const PolyZonotope P = test::random_poly(rng, 2);
        const std::vector<PolyZonotope> D = P.jacobian();
        check(D.size() == 2 && D[0].dim() == 2 && D[0].ids() == P.ids(),
              b + ": one set per factor over the identifiers of the set");
        check(D[0].independent_generators().shape()[0] == 0, b + ": no independent generators");
        check(D[0].num_dependent() == P.num_dependent(), b + ": the columns stay");
        // the exponent of the factor drops by one
        check(D[0].exponents()[2] == std::vector<int>({1, 1}) &&
                  D[1].exponents()[1] == std::vector<int>({0, 0}),
              b + ": the exponents are lowered");
        // the point against a central difference of the dependent part
        const double a1 = 0.4, a2 = -0.8, h = 1e-5;
        for (int k = 0; k < 2; ++k) {
            const std::vector<double> up = test::point(P,
                                                       {{1, a1 + (k == 0 ? h : 0.0)},
                                                        {2, a2 + (k == 1 ? h : 0.0)}},
                                                       {}),
                                      down = test::point(P,
                                                         {{1, a1 - (k == 0 ? h : 0.0)},
                                                          {2, a2 - (k == 1 ? h : 0.0)}},
                                                         {});
            std::vector<double> diff(2);
            for (std::size_t i = 0; i < 2; ++i) diff[i] = (up[i] - down[i]) / (2 * h);
            const std::vector<double> d = test::point(D[std::size_t(k)], {{1, a1}, {2, a2}}, {});
            check(close(Tensor::from_data(d, {2}), Tensor::from_data(diff, {2}), 1e-6),
                  b + ": the derivative along factor " + std::to_string(k + 1));
        }
        // a chosen factor, an unknown one
        check(P.jacobian(std::vector<int64_t>{2}).size() == 1, b + ": a chosen factor");
        check(test::raises([&] { P.jacobian(std::vector<int64_t>{9}); }, Error::wrong_value),
              b + ": an identifier the set lacks");
        // a set without generators
        const PolyZonotope O = PolyZonotope::origin(2);
        check(O.jacobian().empty(), b + ": no factors, no derivatives");
    });
    const PolyZonotope P = test::sample_poly();
    ChecksScope off(false);
    check(P.jacobian().size() == 2, "jacobian: valid, checks off");
    return test::finish("poly_zonotope jacobian");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
