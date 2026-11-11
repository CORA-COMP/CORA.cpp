// test_poly_zonotope_rand_point - PolyZonotope rand_point: random points of the set
//
// CORA: contSet/@polyZonotope/randPoint_.m
// PROPERTY: contSet/@polyZonotope/randPoint_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
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
        Rng rng(1);
        const PolyZonotope P = known();
        const Tensor x = P.rand_point(50, rng);
        check(x.shape() == std::vector<int64_t>({2, 50}), b + ": the shape");
        check(test::inside(P.zonotope(), x), b + ": the points are in the enclosure");
    });

    // a negative count is refused
    {
        Rng bad(1);
        check(test::raises([&] { known().rand_point(-1, bad); }, Error::wrong_value),
              "rand_point: a negative count");
    }
    ChecksScope off(false);
    check(
        [&] {
            Rng r(2);
            return known().rand_point(3, r).shape()[1] == 3;
        }(),
        "rand_point: valid with the checks off");
    return test::finish("poly_zonotope rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
