// test_poly_zonotope_subs - PolyZonotope subs: factors replaced by polynomials
//
// Replacing the factor 2 by the polynomial 0.5 + 0.5 s t of the new factor t gives the set at
// factor 2 = 0.5 + 0.5 s t (a monomial case and a general one), shared factors included.
//
// Oracle debt: the request is in docs/oracle-requests.md (CORALean has the float definition).
//
// CORA: contSet/@polyZonotope/subs.m
// PROPERTY: contSet/@polyZonotope/subs.m
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
        Rng rng(27);
        const PolyZonotope P = test::random_poly(rng, 2);
        // x = 0.5 + 2 t + 0.5 t^2 over the new factor t (id 7)
        const PolyZonotope in(vec({0.5}), Tensor::from_data({2.0, 0.5}, {2, 1}),
                              Tensor::zeros({0, 1}), {{1}, {2}}, {7});
        const PolyZonotope S = P.subs(in, std::vector<int64_t>{2});
        check(S.ids() == std::vector<int64_t>({1, 7}), b + ": the factors of the result");
        const double t = -0.6;
        const double x = 0.5 + 2 * t + 0.5 * t * t;
        check(close(Tensor::from_data(test::point(S, {{1, 0.3}, {7, t}}, {0.2, -0.1}), {2}),
                    Tensor::from_data(test::point(P, {{1, 0.3}, {2, x}}, {0.2, -0.1}), {2}), 1e-10),
              b + ": the substituted point");
        // a factor of the substituted polynomial that the set has already is the same factor
        const PolyZonotope shared(vec({0.0}), Tensor::from_data({1.0}, {1, 1}),
                                  Tensor::zeros({0, 1}), {{1}}, {1});
        const PolyZonotope T = P.subs(shared, std::vector<int64_t>{2});
        check(close(Tensor::from_data(test::point(T, {{1, 0.4}}, {0.2, -0.1}), {2}),
                    Tensor::from_data(test::point(P, {{1, 0.4}, {2, 0.4}}, {0.2, -0.1}), {2}),
                    1e-10),
              b + ": shared factors stay shared");
        // a constant polynomial is a resolve
        const PolyZonotope C(vec({0.25}), Tensor::zeros({0, 1}), Tensor::zeros({0, 1}), {}, {});
        check(P.subs(C, std::vector<int64_t>{2})
                  .isequal(P.resolve(vec({0.25}), std::vector<int64_t>{2}), 1e-12),
              b + ": a constant polynomial");
    });
    const PolyZonotope P = test::sample_poly();
    check(test::raises([&] { P.subs(PolyZonotope::origin(1), std::vector<int64_t>{9}); },
                       Error::wrong_value),
          "subs: an unknown identifier");
    check(test::raises([&] { P.subs(PolyZonotope::origin(3)); }, Error::wrong_value),
          "subs: the dimension of pZin");
    const PolyZonotope with_gi(vec({0.0}), Tensor::zeros({0, 1}), Tensor::from_data({1.0}, {1, 1}),
                               {}, {});
    check(test::raises([&] { P.subs(with_gi, std::vector<int64_t>{2}); }, Error::wrong_value),
          "subs: independent generators");
    ChecksScope off(false);
    check(P.subs(PolyZonotope::origin(2)).dim() == 2, "subs: valid with the checks off");
    return test::finish("poly_zonotope subs");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
