// test_poly_zonotope_is_full_dim - PolyZonotope is_full_dim: whether the set has an interior
//
// A rank below n is false. The independent generators of rank n, or a zonotope as the dependent
// part with rank n, are true. The parabola arc (a, a^2) has rank 2 and no interior: false, where
// CORA says true (docs/cora-differences.md).
//
// CORA: contSet/@polyZonotope/isFullDim.m
// PROPERTY: contSet/@polyZonotope/isFullDim.m
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
        const Tensor I2 = Tensor::from_data({1, 0, 0, 1}, {2, 2});
        // (a, a^2): the exponents 1 and 2 of one factor: rank 2, no interior
        const PolyZonotope parabola(vec({0.0, 0.0}), I2, Tensor::zeros({0, 2}), {{1}, {2}}, {1});
        check(!parabola.is_full_dim(), b + ": the parabola arc has no interior");
        // two factors, one per generator: a box
        const PolyZonotope box(vec({0.0, 0.0}), I2, Tensor::zeros({0, 2}), {{1, 0}, {0, 1}},
                               {1, 2});
        check(box.is_full_dim(), b + ": a zonotope of full rank");
        // independent generators of rank n
        const PolyZonotope indep(vec({0.0, 0.0}), Tensor::from_data({1, 1}, {1, 2}), I2, {{1}},
                                 {1});
        check(indep.is_full_dim(), b + ": the independent generators span the space");
        // rank 1
        const PolyZonotope flat(vec({0.0, 0.0}), Tensor::from_data({1, 1}, {1, 2}),
                                Tensor::from_data({2, 2}, {1, 2}), {{1}}, {1});
        check(!flat.is_full_dim(), b + ": a segment");
        check(!PolyZonotope::origin(2).is_full_dim(), b + ": a point");
        // a random set with two independent generators in the plane
        Rng rng(4);
        check(test::random_poly(rng, 2).is_full_dim(), b + ": random independent generators");
    });
    check(test::raises([] { PolyZonotope::origin(2).is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol < 0");
    ChecksScope off(false);
    check(!PolyZonotope::origin(2).is_full_dim(), "is_full_dim: valid with the checks off");
    return test::finish("poly_zonotope is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
