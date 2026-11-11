// test_poly_zonotope_exact_plus - PolyZonotope exact_plus: the sum with shared factors
//
// CORA: contSet/@polyZonotope/exactPlus.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const PolyZonotope P0(P.center(), P.generators(), Tensor::zeros({2, 0}), P.exponents(),
                              P.ids());
        // x + (-x) with the same factors is the origin: the dependency stays
        const PolyZonotope Z = P0.exact_plus(P0.linear_map(Tensor::eye(2) * -1.0));
        check(Z.num_dependent() == 0 && close(Z.center(), {0.0, 0.0}, 0),
              b + ": x - x = 0 in the dependent part");
        const PolyZonotope W = P.plus(P.linear_map(Tensor::eye(2) * -1.0));
        check(W.num_dependent() == 4, b + ": plus does not know the factors are shared");
        // a set with other identifiers is added with the union of the identifiers
        const PolyZonotope Q(column({0.0, 0.0}), Tensor::from_data({1, 1}, {2, 1}),
                             Tensor::zeros({2, 0}), {{1}}, {3});
        check(P.exact_plus(Q).num_factors() == 3, b + ": the union of the identifiers");
    });
    check(test::raises(
              [] {
                  known().exact_plus(PolyZonotope(Zonotope(column({1.0}), Tensor::zeros({1, 0}))));
              },
              Error::dimension_mismatch),
          "exact_plus: dimension");
    ChecksScope off(false);
    check(known().exact_plus(known()).dim() == 2, "exact_plus: valid with the checks off");
    return test::finish("poly_zonotope exact_plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
