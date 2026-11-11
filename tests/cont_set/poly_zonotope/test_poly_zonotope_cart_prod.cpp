// test_poly_zonotope_cart_prod - PolyZonotope cart_prod: the product with a zonotope
//
// CORA: contSet/@polyZonotope/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
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
        const Zonotope Z(column({5.0}), Tensor::from_data({1.0, 2.0}, {1, 2}));
        const PolyZonotope Q = P.cart_prod(Z);
        check(Q.dim() == 3 && close(Q.center(), {1.0, 0.0, 5.0}, 0), b + ": the centers stack");
        check(Q.independent_generators().shape()[1] == 3 && Q.num_dependent() == 2,
              b + ": the zonotope is independent");
        check(cart_prod(P, Z).is<PolyZonotope>(), b + ": through the registry");
    });

    check(
        test::raises(
            [] { known().cart_prod(Zonotope(Tensor::zeros({2, 1, 1}), Tensor::zeros({2, 1, 0}))); },
            Error::dimension_mismatch),
        "cart_prod: a batch");
    ChecksScope off(false);
    check(known().cart_prod(Zonotope(column({1.0}), Tensor::zeros({1, 0}))).dim() == 3,
          "cart_prod: valid with the checks off");
    return test::finish("poly_zonotope cart_prod");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
