// test_poly_zonotope_linear_map - PolyZonotope linear_map: a matrix and an interval matrix
//
// CORA: contSet/@polyZonotope/mtimes.m
// PROPERTY: contSet/@polyZonotope/mtimes.m
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

/// Stated property: the image of a point under a matrix of the interval lies in the result.
void map_property(const std::string &b) {
    Rng rng(9);
    for (int k = 0; k < 5; ++k) {
        const PolyZonotope P = test::random_poly(rng, 3);
        const Tensor M = test::normal(rng, 3, 3),
                     R = Tensor::from_data(std::vector<double>(9, 0.1), {3, 3});
        const Tensor x = test::sample(P, rng, 20);
        check(test::inside(P.linear_map(M).zonotope(), M.matmul(x)), b + ": M x is inside");
        check(
            test::inside(P.linear_map(Interval(M - R, M + R)).zonotope(), (M + R * 0.5).matmul(x)),
            b + ": a matrix of the interval maps inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        const PolyZonotope Q = P.linear_map(Tensor::from_data({2, 0, 0, 1}, {2, 2}));
        check(close(Q.center(), {2.0, 0.0}, 0) && close(Q.generators(), {2.0, 0.0, 1.0, 2.0}, 0) &&
                  Q.exponents() == P.exponents(),
              b + ": exact, the exponents stay");
        const Interval M(Tensor::eye(2) * 0.9, Tensor::eye(2) * 1.1);
        check(P.linear_map(M).independent_generators().shape()[1] == 3,
              b + ": the width joins the independent part");
        map_property(b);
    });
    check(
        test::raises([] { known().linear_map(Tensor::zeros({2, 3})); }, Error::dimension_mismatch),
        "linear_map: columns");
    ChecksScope off(false);
    check(known().linear_map(Tensor::eye(2)).dim() == 2, "linear_map: valid with the checks off");
    return test::finish("poly_zonotope linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
