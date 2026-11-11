// test_poly_zonotope_enclose - PolyZonotope enclose: the hull of two polynomial zonotopes
//
// CORA: contSet/@polyZonotope/enclose.m
// PROPERTY: contSet/@polyZonotope/enclose.m
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

/// Stated property: the points between a point of each set lie in the enclosure, with the same
/// and with other factors.
void enclose_property(const std::string &b) {
    Rng rng(21);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope A = test::random_poly(rng, 3), B = test::random_poly(rng, 3);
        const Tensor mid = (test::sample(A, rng, 20) + test::sample(B, rng, 20)) * 0.5;
        check(test::inside(A.enclose(B).zonotope(), mid), b + ": the midpoints are inside");
        const PolyZonotope C = A.linear_map(Tensor::eye(3) * 2.0);
        check(test::inside(A.enclose(C).zonotope(), test::sample(A, rng, 20)),
              b + ": same factors, a point of A");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const PolyZonotope O = P.linear_map(Tensor::zeros({2, 2}));
        // the hull of P and 0 (with the factors of P): the points lambda x
        const PolyZonotope H = P.enclose(O);
        check(H.num_factors() == 3, b + ": one new factor");
        check(close(H.center(), {0.5, 0.0}, 1e-12), b + ": the midpoint of the centers");
        enclose_property(b);
    });
    check(test::raises(
              [] { known().enclose(PolyZonotope(Zonotope(column({1.0}), Tensor::zeros({1, 0})))); },
              Error::dimension_mismatch),
          "enclose: dimension");
    ChecksScope off(false);
    check(known().enclose(known()).dim() == 2, "enclose: valid with the checks off");
    return test::finish("poly_zonotope enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
